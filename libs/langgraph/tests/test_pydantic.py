import datetime
import decimal
import ipaddress
import pathlib
import re
import sys
import typing
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from threading import Event
from typing import Annotated, Literal, Optional

import pydantic
import pytest
import typing_extensions
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import (
    BaseModel,
    ByteSize,
    ConfigDict,
    Field,
    SecretStr,
    confloat,
    conint,
    conlist,
    constr,
    field_validator,
    model_validator,
)

from langgraph._internal._pydantic import create_model, is_supported_by_pydantic
from langgraph.constants import END, START
from langgraph.errors import InvalidUpdateError
from langgraph.func import entrypoint, task
from langgraph.graph.state import StateGraph
from langgraph.types import Command, Interrupt, interrupt
from tests.any_str import AnyStr


@pytest.mark.parametrize("root_model", [False, True], ids=["fields", "root"])
@pytest.mark.parametrize("default", [None, []], ids=["cached", "uncached"])
def test_create_model_preserves_concurrent_warnings(root_model, default) -> None:
    schema_started = Event()
    warning_emitted = Event()

    class PausedSchema:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            schema_started.set()
            assert warning_emitted.wait(timeout=10)
            return handler(source)

    type_ = Annotated[list[int] | None, PausedSchema]
    kwargs = (
        {"root": (type_, default)}
        if root_model
        else {"field_definitions": {"model_values": (type_, default)}}
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(create_model, "ConcurrentSchema", **kwargs)
            try:
                assert schema_started.wait(timeout=10)
                warnings.warn("unrelated concurrent warning", UserWarning)
                warnings.simplefilter("error", DeprecationWarning)
            finally:
                warning_emitted.set()
            model = future.result(timeout=10)

        assert [str(warning.message) for warning in captured] == [
            "unrelated concurrent warning"
        ]
        with pytest.raises(DeprecationWarning, match="retained warning filter"):
            warnings.warn("retained warning filter", DeprecationWarning)

    instance = model.model_validate(["1"] if root_model else {"model_values": ["1"]})
    assert instance.model_dump() == ([1] if root_model else {"model_values": [1]})


@pytest.mark.parametrize("default", [0, []], ids=["cached", "uncached"])
def test_create_model_model_fields(default) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = create_model(
            "ModelFields",
            field_definitions={
                "model_id": (int, ...),
                "model_name": (str, "default"),
                "model_dump": (str, ...),
                "_private": (int, 0),
                "model_value": (int | list[int], default),
            },
        )

    instance = model.model_validate(
        {"model_id": "1", "model_dump": "data", "_private": "2"}
    )
    assert model.__name__ == "ModelFields"
    assert instance.model_dump(by_alias=True) == {
        "model_id": 1,
        "model_name": "default",
        "model_dump": "data",
        "_private": 2,
        "model_value": default,
    }
    with pytest.raises(pydantic.ValidationError, match="model_id"):
        model.model_validate({"model_id": "invalid", "model_dump": "data"})
    with pytest.raises(pydantic.ValidationError, match="frozen_instance"):
        instance.model_id = 2

    schema = model.model_json_schema()
    assert schema["title"] == "ModelFields"
    assert schema["required"] == ["model_id", "model_dump"]
    assert set(schema["properties"]) == {
        "model_id",
        "model_name",
        "model_dump",
        "_private",
        "model_value",
    }
    assert schema["properties"]["model_dump"] == {
        "title": "Model Dump",
        "type": "string",
    }
    assert schema["properties"]["model_value"]["default"] == default


@pytest.mark.parametrize("default", [0, []], ids=["cached", "uncached"])
def test_create_root_model_defaults(default) -> None:
    model = create_model("RootSchema", root=(int | list[int], default))
    assert model.__name__ == "RootSchema"
    assert model().model_dump() == default
    assert model.model_validate("2").model_dump() == 2
    schema = model.model_json_schema()
    assert schema == {
        "anyOf": [{"type": "integer"}, {"items": {"type": "integer"}, "type": "array"}],
        "default": default,
        "title": "RootSchema",
    }
    with pytest.warns(pydantic.PydanticDeprecatedSince20, match="deprecated"):
        assert model.schema() == schema


def test_create_model_preserves_schema_warnings() -> None:
    class WarningSchema:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            warnings.warn("custom schema warning", UserWarning)
            return handler(source)

    with pytest.warns(UserWarning, match="custom schema warning"):
        create_model(
            "WarningSchema", field_definitions={"model_value": (WarningSchema, ...)}
        )


def test_is_supported_by_pydantic() -> None:
    """Test if types are supported by pydantic."""

    class TypedDictExtensions(typing_extensions.TypedDict):
        x: int

    assert is_supported_by_pydantic(TypedDictExtensions) is True

    class VanillaClass:
        x: int

    assert is_supported_by_pydantic(VanillaClass) is False

    class BuiltinTypedDict(typing.TypedDict):  # noqa: TID251
        x: int

    if sys.version_info >= (3, 12):
        assert is_supported_by_pydantic(BuiltinTypedDict) is True
    else:
        assert is_supported_by_pydantic(BuiltinTypedDict) is False

    class PydanticModel(pydantic.BaseModel):
        x: int

    assert is_supported_by_pydantic(PydanticModel) is True


def test_nested_pydantic_models() -> None:
    """Test that nested Pydantic models are properly constructed from leaf nodes up."""

    class NestedModel(BaseModel):
        value: int
        name: str

    # For constrained types
    PositiveInt = Annotated[int, Field(gt=0)]
    NonNegativeFloat = Annotated[float, Field(ge=0)]

    # Enum type
    class UserRole(Enum):
        ADMIN = "admin"
        USER = "user"
        GUEST = "guest"

    # Forward reference model
    class RecursiveModel(BaseModel):
        value: str
        child: Optional["RecursiveModel"] = None

    # Discriminated union models
    class Cat(BaseModel):
        pet_type: Literal["cat"]
        meow: str

    class Dog(BaseModel):
        pet_type: Literal["dog"]
        bark: str

    # Cyclic reference model
    class Person(BaseModel):
        id: str
        name: str
        friends: list[str] = Field(default_factory=list)  # IDs of friends

    conlist_type = conlist(item_type=int, min_length=2, max_length=5)

    class State(BaseModel):
        # Basic nested model tests
        top_level: str
        auuid: uuid.UUID
        nested: NestedModel
        optional_nested: Annotated[NestedModel | None, lambda x, y: y, "Foo"]
        dict_nested: dict[str, NestedModel]
        simple_str_list: list[str]
        list_nested: Annotated[
            dict | list[dict[str, NestedModel]], lambda x, y: (x or []) + [y]
        ]
        tuple_nested: tuple[str, NestedModel]
        tuple_list_nested: list[tuple[int, NestedModel]]
        complex_tuple: tuple[str, dict[str, tuple[int, NestedModel]]]

        # Forward reference test
        recursive: RecursiveModel

        # Discriminated union test
        pet: Cat | Dog

        # Cyclic reference test
        people: dict[str, Person]  # Map of ID -> Person

        # Rich type adapters
        ip_address: ipaddress.IPv4Address
        ip_address_v6: ipaddress.IPv6Address
        amount: decimal.Decimal
        file_path: pathlib.Path
        timestamp: datetime.datetime
        date_only: datetime.date
        time_only: datetime.time
        duration: datetime.timedelta
        immutable_set: frozenset[int]
        binary_data: bytes
        pattern: re.Pattern
        secret: SecretStr
        file_size: ByteSize

        # Constrained types
        positive_value: PositiveInt
        non_negative: NonNegativeFloat
        limited_string: constr(min_length=3, max_length=10)
        bounded_int: conint(ge=10, le=100)
        restricted_float: confloat(gt=0, lt=1)
        required_list: conlist_type

        # Enum & Literal
        role: UserRole
        status: Literal["active", "inactive", "pending"]

        # Annotated & NewType
        validated_age: Annotated[int, Field(gt=0, lt=120)]

        # Generic containers with validators
        decimal_list: list[decimal.Decimal]
        id_tuple: tuple[uuid.UUID, uuid.UUID]

    inputs = {
        # Basic nested models
        "top_level": "initial",
        "auuid": str(uuid.uuid4()),
        "nested": {"value": 42, "name": "test"},
        "optional_nested": {"value": 10, "name": "optional"},
        "dict_nested": {"a": {"value": 5, "name": "a"}},
        "list_nested": [{"a": {"value": 6, "name": "b"}}],
        "tuple_nested": ["tuple-key", {"value": 7, "name": "tuple-value"}],
        "tuple_list_nested": [[1, {"value": 8, "name": "tuple-in-list"}]],
        "simple_str_list": ["siss", "boom", "bah"],
        "complex_tuple": [
            "complex",
            {"nested": [9, {"value": 10, "name": "deep"}]},
        ],
        # Forward reference
        "recursive": {"value": "parent", "child": {"value": "child", "child": None}},
        # Discriminated union (using a cat in this case)
        "pet": {"pet_type": "cat", "meow": "meow!"},
        # Cyclic references
        "people": {
            "1": {
                "id": "1",
                "name": "Alice",
                "friends": ["2", "3"],  # Alice is friends with Bob and Charlie
            },
            "2": {
                "id": "2",
                "name": "Bob",
                "friends": ["1"],  # Bob is friends with Alice
            },
            "3": {
                "id": "3",
                "name": "Charlie",
                "friends": ["1", "2"],  # Charlie is friends with Alice and Bob
            },
        },
        # Rich type adapters
        "ip_address": "192.168.1.1",
        "ip_address_v6": "2001:db8::1",
        "amount": "123.45",
        "file_path": "/tmp/test.txt",
        "timestamp": "2025-04-07T10:58:04",
        "date_only": "2025-04-07",
        "time_only": "10:58:04",
        "duration": 3600,  # seconds
        "immutable_set": [1, 2, 3, 4],
        "binary_data": b"hello world",
        "pattern": "^test$",
        "secret": "password123",
        "file_size": 1024,
        # Constrained types
        "positive_value": 42,
        "non_negative": 0.0,
        "limited_string": "test",
        "bounded_int": 50,
        "restricted_float": 0.5,
        "required_list": [10, 20, 30],
        # Enum & Literal
        "role": "admin",
        "status": "active",
        # Annotated & NewType
        "validated_age": 30,
        # Generic containers with validators
        "decimal_list": ["10.5", "20.75", "30.25"],
        "id_tuple": [str(uuid.uuid4()), str(uuid.uuid4())],
    }

    update = {"top_level": "updated", "nested": {"value": 100, "name": "updated"}}

    expected = State(**inputs)

    def node_fn(state: State) -> dict:
        # Basic assertions
        assert isinstance(state.auuid, uuid.UUID)
        assert state == expected

        # Rich type assertions
        assert isinstance(state.ip_address, ipaddress.IPv4Address)
        assert isinstance(state.ip_address_v6, ipaddress.IPv6Address)
        assert isinstance(state.amount, decimal.Decimal)
        assert isinstance(state.file_path, pathlib.Path)
        assert isinstance(state.timestamp, datetime.datetime)
        assert isinstance(state.date_only, datetime.date)
        assert isinstance(state.time_only, datetime.time)
        assert isinstance(state.duration, datetime.timedelta)
        assert isinstance(state.immutable_set, frozenset)
        assert isinstance(state.binary_data, bytes)
        assert isinstance(state.pattern, re.Pattern)

        # Constrained types
        assert state.positive_value > 0
        assert state.non_negative >= 0
        assert 3 <= len(state.limited_string) <= 10
        assert 10 <= state.bounded_int <= 100
        assert 0 < state.restricted_float < 1
        assert 2 <= len(state.required_list) <= 5

        # Enum & Literal
        assert state.role == UserRole.ADMIN
        assert state.status == "active"

        # Annotated
        assert 0 < state.validated_age < 120

        # Generic containers
        assert len(state.decimal_list) == 3
        assert len(state.id_tuple) == 2

        return update

    builder = StateGraph(State)
    builder.add_node("process", node_fn)
    builder.set_entry_point("process")
    builder.set_finish_point("process")
    graph = builder.compile()

    result = graph.invoke(inputs.copy())

    assert result == {**inputs, **update}

    new_inputs = inputs.copy()
    new_inputs["list_nested"] = {"foo": "bar"}
    expected = State(**new_inputs)
    assert {**new_inputs, **update} == graph.invoke(new_inputs.copy())


def test_pydantic_state_field_validator():
    class State(BaseModel):
        name: str
        text: str = ""
        only_root: int = 13

        @field_validator("name", mode="after")
        @classmethod
        def validate_name(cls, value):
            if value[0].islower():
                raise ValueError("Name must start with a capital letter")
            return "Validated " + value

        @model_validator(mode="before")
        @classmethod
        def validate_amodel(cls, values: "State"):
            return values | {"only_root": 392}

    input_state = {"name": "John"}

    def process_node(state: State):
        assert State.model_validate(input_state) == state
        return {"text": "Hello, " + state.name + "!"}

    builder = StateGraph(state_schema=State)
    builder.add_node("process", process_node)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    g = builder.compile()
    res = g.invoke(input_state)
    assert res["text"] == "Hello, Validated John!"


class FunctionalState(BaseModel):
    a: str
    b: str | None = None


def test_interrupt_functional_pydantic(sync_checkpointer: BaseCheckpointSaver) -> None:
    called_count = 0

    @task
    def foo(state: FunctionalState) -> FunctionalState:
        nonlocal called_count
        called_count += 1
        return FunctionalState(**{"a": state.a + "foo"})

    @task
    def bar(state: FunctionalState) -> dict:
        return {"a": state.a + "bar", "b": state.b}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(inputs: FunctionalState) -> FunctionalState:
        fut_foo = foo(inputs)
        value = interrupt("Provide value for bar:")
        foo_res = fut_foo.result()
        assert isinstance(foo_res, FunctionalState)
        bar_input = FunctionalState(a=foo_res.a, b=value)
        fut_bar = bar(bar_input)
        return fut_bar.result()

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke(FunctionalState(a=""), config) == {
        "__interrupt__": [
            Interrupt(
                value="Provide value for bar:",
                id=AnyStr(),
            )
        ]
    }
    # Resume with an answer
    res = graph.invoke(Command(resume="bar"), config)
    assert res == {"a": "foobar", "b": "bar"}
    assert called_count == 1


def test_aliased_pydantic_graph_input_and_checkpoint() -> None:
    class State(BaseModel):
        foo: str = Field(alias="bar")

        @field_validator("foo")
        @classmethod
        def validate_foo(cls, value: str) -> str:
            if value == "invalid":
                raise ValueError("invalid foo")
            return value

    seen = []

    def node(state: State):
        seen.append(state.foo)
        return None

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "alias-test"}}
    assert graph.invoke({"bar": "hello"}, config) == {"foo": "hello"}
    assert seen == ["hello"]
    assert graph.get_state(config).values == {"foo": "hello"}
    with pytest.raises(pydantic.ValidationError, match="invalid foo"):
        graph.invoke({"bar": "invalid"}, {"configurable": {"thread_id": "invalid"}})


def test_pydantic_shared_alias_preserves_canonical_state() -> None:
    class State(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        a: int = Field(alias="shared")
        b: int = Field(alias="shared")

    seen = []

    def node(state: State):
        seen.append(state.model_dump())
        return None

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    assert builder.compile().invoke({"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert seen == [{"a": 1, "b": 2}]


def test_aliased_pydantic_input_respects_name_validation() -> None:
    class AliasOnly(BaseModel):
        foo: str = Field(alias="bar")

    class AliasOrName(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        foo: str = Field(alias="bar")

    class NameOnly(BaseModel):
        model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)
        foo: str = Field(alias="bar")

    def graph_for(schema):
        builder = StateGraph(schema)
        builder.add_node("node", lambda state: None)
        builder.add_edge(START, "node")
        return builder.compile()

    with pytest.raises(pydantic.ValidationError, match="bar"):
        graph_for(AliasOnly).invoke({"foo": "hello"})
    assert graph_for(AliasOnly).invoke({"foo": "name", "bar": "alias"}) == {
        "foo": "alias"
    }
    assert graph_for(AliasOrName).invoke({"foo": "hello"}) == {"foo": "hello"}
    if "validate_by_alias" in ConfigDict.__annotations__:
        assert graph_for(NameOnly).invoke({"foo": "hello"}) == {"foo": "hello"}
        with pytest.raises(pydantic.ValidationError, match="foo"):
            graph_for(NameOnly).invoke({"bar": "hello"})
    else:
        # Pydantic versions before 2.11 ignore the newer validation options.
        assert graph_for(NameOnly).invoke({"bar": "hello"}) == {"foo": "hello"}
        with pytest.raises(pydantic.ValidationError, match="bar"):
            graph_for(NameOnly).invoke({"foo": "hello"})
    assert graph_for(AliasOrName).invoke({"foo": "name", "bar": "alias"}) == {
        "foo": "alias"
    }


def test_defaulted_alias_only_field_keeps_checkpoint_keys_canonical() -> None:
    class State(BaseModel):
        foo: int = Field(default=0, alias="wire")

    seen = []

    def node(state: State):
        seen.append(state.foo)
        return None

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "defaulted-alias-field"}}

    assert graph.invoke({"foo": 9}, config) is None
    assert seen == [0]
    assert graph.get_state(config).values == {}
    checkpoint = saver.get(config)
    assert checkpoint is not None
    assert "foo" not in checkpoint["channel_values"]
    assert "wire" not in checkpoint["channel_values"]

    assert graph.invoke({"wire": 7}, config) == {"foo": 7}
    assert graph.get_state(config).values == {"foo": 7}

    assert graph.invoke({"foo": 9}, config) is None
    assert seen == [0, 7, 7]
    assert graph.get_state(config).values == {"foo": 7}
    checkpoint = saver.get(config)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["foo"] == 7
    assert "wire" not in checkpoint["channel_values"]


def test_aliased_subgraph_accepts_canonical_parent_state() -> None:
    class ChildState(BaseModel):
        foo: int = Field(alias="wire")

    class ParentState(typing_extensions.TypedDict):
        foo: int

    child = StateGraph(ChildState)
    child.add_node("increment", lambda state: {"foo": state.foo + 1})
    child.add_edge(START, "increment")

    parent = StateGraph(ParentState)
    parent.add_node("child", child.compile())
    parent.add_edge(START, "child")
    assert parent.compile().invoke({"foo": 1}) == {"foo": 2}


def test_aliased_input_schema_keeps_canonical_channels() -> None:
    class Input(BaseModel):
        foo: str = Field(alias="bar")

    class State(typing_extensions.TypedDict):
        foo: str

    seen = []

    def node(state: State):
        seen.append(state["foo"])
        return None

    builder = StateGraph(State, input_schema=Input)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    assert builder.compile().invoke({"bar": "hello"}) == {"foo": "hello"}
    assert seen == ["hello"]


def test_aliased_pydantic_separate_schemas_and_stream() -> None:
    class Input(BaseModel):
        foo: str = Field(alias="request_foo")

    class State(BaseModel):
        foo: str = Field(alias="state_foo")

    class Output(BaseModel):
        foo: str = Field(alias="response_foo")

    observed = []

    def node(state: State):
        observed.append(state.foo)
        return None

    builder = StateGraph(State, input_schema=Input, output_schema=Output)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    graph = builder.compile()
    assert list(graph.stream({"request_foo": "hello"}, stream_mode="values")) == [
        {"foo": "hello"}
    ]
    assert observed == ["hello"]
    assert graph.invoke({"request_foo": "hello"}, version="v2").value.foo == "hello"


def test_aliased_pydantic_alias_generator() -> None:
    class State(BaseModel):
        model_config = ConfigDict(alias_generator=str.upper)
        foo: str

    builder = StateGraph(State)
    builder.add_node("node", lambda state: None)
    builder.add_edge(START, "node")
    assert builder.compile().invoke({"FOO": "hello"}) == {"foo": "hello"}


@pytest.mark.anyio
async def test_aliased_pydantic_async_and_resume() -> None:
    class State(BaseModel):
        foo: str = Field(alias="bar")

    seen = []

    def node(state: State):
        seen.append(state.foo)
        return None

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["node"])
    config = {"configurable": {"thread_id": "async-alias"}}
    events = [
        event
        async for event in graph.astream(
            {"foo": "ignored", "bar": "hello"}, config, stream_mode="values"
        )
    ]
    assert events == [{"foo": "hello"}]
    assert (await graph.aget_state(config)).values == {"foo": "hello"}
    await graph.ainvoke(None, config)
    assert seen == ["hello"]


def test_aliased_pydantic_name_and_before_validator_preserves_existing_behavior() -> (
    None
):
    class State(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        foo: int = Field(alias="bar")

        @model_validator(mode="before")
        @classmethod
        def validate_input(cls, value):
            if "foo" in value:
                return value | {"foo": value["foo"] + 1}
            return value

    seen = []

    def node(state: State):
        seen.append(state.foo)
        return None

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    graph = builder.compile()
    assert graph.invoke({"foo": 1}) == {"foo": 1}
    assert seen == [2]
    with pytest.raises(InvalidUpdateError, match="bar"):
        graph.invoke({"bar": 1})


def test_aliased_checkpoint_stream_ignores_other_channel_collision() -> None:
    class State(BaseModel):
        foo: int = Field(alias="bar")

    class Extra(typing_extensions.TypedDict):
        bar: int

    builder = StateGraph(State)
    builder.add_node("extra", lambda state: {"bar": 99}, input_schema=Extra)
    builder.add_edge(START, "extra")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "alias-collision"}}
    graph.invoke({"bar": 5}, config)
    assert graph.get_state(config).values == {"foo": 5, "bar": 99}
    events = list(graph.stream(None, config, stream_mode="checkpoints", version="v2"))
    values = [
        event["data"]["values"] for event in events if event["data"].get("values")
    ]
    assert values
    assert all(value.foo == 5 for value in values)
