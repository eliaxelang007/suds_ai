from typing import Union, Literal, TypedDict, Callable, Iterable, Annotated, get_origin, get_args, Any, cast
from inspect import signature, Parameter
from types import NoneType, UnionType

type Number = int | float
type Value = None | bool | Number | str | tuple["Value", ...] | dict[str, "Value"]
type ValueType = Literal["null", "boolean", "number", "string", "array", "object"]

class JsonSchemaType[T: ValueType](TypedDict):
    type: T
    description: str

class _JsonSchemaEnumTemplate[T: ValueType, S: Value](JsonSchemaType[T]):
    enum: tuple[S]

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

    def to_json_schema(parameter_annotation: TypeExpr | None) -> JsonSchema:
        assert get_origin(
            parameter_annotation
        ) is Annotated, f"The parameter type must be wrapped in [{Annotated}] with its annotation being the parameter's description."

        parameter_type, description = get_args(parameter_annotation)

        assert isinstance(description, str), "The parameter's description must be a string!"

        def type_to_json_type(from_type: TypeExpr | None) -> ValueType | None:
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

        arguments: set[Value] = set()
        argument_types: set[ValueType] = set()

        for argument in traverse_union(parameter_type):
            argument = argument if argument != NoneType else None
            argument_json_type = type_to_json_type(type(argument))

            assert (
                argument_json_type is not None
            ), f"Unsupported non-literal union annotation [{parameter_type}]!"

            arguments.add(argument)
            argument_types.add(argument_json_type)

            assert (
                (len(argument_types) == 1) or (len(argument_types) == 2 and "null" in argument_types)
            ), f"Unions with literals of different types like [{parameter_type}] aren't supported!"

        assert (
            len(argument_types) >= 1
        ), f"Unsupported type annotation [{parameter_type}]!"

        return cast(JsonSchemaEnumType, {
            "type": tuple(argument_types) if len(argument_types) > 1 else argument_types.pop(),
            "enum": tuple(arguments),
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
            "description": description.strip(),
            "parameters": {
                "type": "object",
                "properties": parameters
            },
            "required": required,
            "additionalProperties": False
        }
    }