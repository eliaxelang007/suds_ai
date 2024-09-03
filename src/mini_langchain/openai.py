from inspect import signature, Parameter
from typing import Union, Literal, TypedDict, Callable, Iterable, Annotated, get_origin, get_args, Any, cast
from types import NoneType, UnionType

from .utils.strings import clean_multi_line

type Number = int | float
type Value = None | bool | Number | str | tuple["Value", ...] | dict[str, "Value"]
type ValueType = Literal["null", "boolean", "number", "string", "array", "object"]

class JsonSchemaType[T: ValueType](TypedDict):
    type: T | tuple[T, Literal["null"]]
    description: str

class _JsonSchemaEnumTemplate[T: ValueType, S: Value](JsonSchemaType[T]):
    enum: tuple[S, ...] | tuple[None, *tuple[S, ...]]

type JsonSchemaEnumType = (
    _JsonSchemaEnumTemplate[Literal["string"], str] |
    _JsonSchemaEnumTemplate[Literal["number"], Number] |
    _JsonSchemaEnumTemplate[Literal["null"], None] |
    _JsonSchemaEnumTemplate[Literal["boolean"], bool] |
    _JsonSchemaEnumTemplate[Literal["array"], tuple[Value, ...]] |
    _JsonSchemaEnumTemplate[Literal["object"], dict[str, Value]]
)

type JsonSchema = JsonSchemaType[ValueType] | JsonSchemaEnumType

class OpenaiFunctionDefinitionParameters(TypedDict):
    type: Literal["object"]
    properties: dict[str, JsonSchema]

class OpenaiFunctionDefinition(TypedDict):
    name: str
    strict: Literal[True]
    description: str
    parameters: OpenaiFunctionDefinitionParameters
    required: list[str]
    additionalProperties: Literal[False]

class OpenaiToolDefinition(TypedDict):
    type: Literal["function"]
    function: OpenaiFunctionDefinition

def to_openai_tool(function: Callable[..., str]) -> OpenaiToolDefinition:
    # Terrible workaround for 
    # https://stackoverflow.com/questions/78814678/if-union-has-uniontype-whats-the-uniontype-equivalent-of-literal
    type TypeExpr = Any 

    def to_json_schema(parameter_annotation: TypeExpr | type | None) -> JsonSchema:
        assert get_origin(
            parameter_annotation
        ) is Annotated, f"The parameter type must be wrapped in [{Annotated}] with its annotation being the parameter's description."

        parameter_type, description = get_args(parameter_annotation)

        assert isinstance(description, str), "The parameter's description must be a string!"

        def type_to_json_type(from_type: TypeExpr | type | None) -> ValueType | None:
            SCHEMA_TYPE_MAPPING: dict[type | None, ValueType] = {
                str: "string",
                int: "number",
                float: "number",
                dict: "object",
                list: "array",
                bool: "boolean",
                NoneType: "null",
                None: "null"
            }

            return SCHEMA_TYPE_MAPPING.get(from_type, None)

        parameter_json_type = type_to_json_type(parameter_type)

        if parameter_json_type is not None:
            return {
                "type" : parameter_json_type,
                "description": description
            }
        
        def traverse_union(union_type: TypeExpr) -> Iterable[Any]:
            def is_union_type(maybe_union_type: TypeExpr) -> bool:
                return (
                    get_origin(maybe_union_type) in 
                    {UnionType, Union, Literal}
                )
            
            if not is_union_type(union_type):
                EMPTY_ITERABLE = iter(())
                return EMPTY_ITERABLE

            def _traverse_union(union_type: TypeExpr) -> Iterable[Any]:
                if is_union_type(union_type):
                    for argument in get_args(union_type):
                        yield from _traverse_union(argument)
                    return
                
                yield union_type

            return _traverse_union(union_type)

        enum_variants_type: ValueType | None = None
        is_optional = False

        enum_variants: set[Value] = set()

        for argument in traverse_union(parameter_type):
            argument_is_type = isinstance(argument, type)
            argument_type = argument if argument_is_type else type(argument)

            argument_json_type = type_to_json_type(argument_type)

            assert (
                argument_json_type is not None
            ), f"Unsupported type [{argument_type}]!"

            enum_variants_type = (
                enum_variants_type if 
                (
                    enum_variants_type is not None and 
                    enum_variants_type != "null"
                ) else
                argument_json_type
            )

            argument_is_optional = argument_json_type == "null"

            assert (
                argument_json_type == enum_variants_type or argument_is_optional
            ), f"Unsupported non-literal union annotation [{parameter_type}]!"

            is_optional = is_optional or argument_is_optional

            if not argument_is_type:
                enum_variants.add(argument)

        assert (
            enum_variants_type is not None
        ), f"Unsupported type [{parameter_type}]!"

        if is_optional:
            enum_variants.add(None)

        return cast(JsonSchema, {
            "type": (enum_variants_type, "null") if is_optional else enum_variants_type,
            **(
                {"enum": tuple(enum_variants)} if 
                len(enum_variants) >= 1 else 
                {}
            ),
            "description": description
        })
    
    function_signature = signature(function)
    
    assert (
        issubclass(function_signature.return_annotation, str)
    ), "The return type of the function must be a string!"

    parameters: dict[str, JsonSchema] = {}
    required: list[str] = []

    for parameter_name, parameter_type in function_signature.parameters.items():
        assert (
            parameter_type.annotation != Parameter.empty
        ), "Function parameter has no type annotation!"

        parameters[parameter_name] = to_json_schema(parameter_type.annotation)
        
        if parameter_type.default == Parameter.empty:
            required.append(parameter_name)

    description = function.__doc__
    assert description is not None, "Function has no docstring description!"

    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "strict": True,
            "description": clean_multi_line(description),
            "parameters": {
                "type": "object",
                "properties": parameters
            },
            "required": required,
            "additionalProperties": False
        }
    }