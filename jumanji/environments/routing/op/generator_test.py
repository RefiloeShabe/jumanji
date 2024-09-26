import chex
import jax
import pytest

from jumanji.environments.routing.op.conftest import DummyGenerator
from jumanji.environments.routing.op.generator import UniformGenerator
from jumanji.environments.routing.op.generator import ConstantGenerator
from jumanji.environments.routing.op.generator import ProportionalGenerator
from jumanji.environments.routing.op.types import State
from jumanji.testing.pytrees import assert_trees_are_different, assert_trees_are_equal


class TestDummyGenerator:
    @pytest.fixture
    def dummy_generator(self) -> DummyGenerator:
        return DummyGenerator()

    def test_dummy_generator__properties(self, dummy_generator: DummyGenerator) -> None:
        """Validate that the dummy instance generator has the correct properties."""
        assert dummy_generator.num_nodes == 4
        assert dummy_generator.max_length == 0.4

    def test_dummy_generator__call(self, dummy_generator: DummyGenerator) -> None:
        """Validate that the dummy instance generator's call function behaves correctly,
        that it is jit-table and compiles only once, and that it returns the same state
        for different keys
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(dummy_generator.__call__, n=1))
        state1 = call_fn(jax.random.PRNGKey(1))
        state2 = call_fn(jax.random.PRNGKey(2))
        assert_trees_are_equal(state1, state2)


class TestConstantGenerator:
    @pytest.fixture
    def constant_generator(self) -> ConstantGenerator:
        return ConstantGenerator(
            num_nodes=20,
            max_length=2,
        )

    def test_constant_generator__properties(
        self, constant_generator: ConstantGenerator
    ) -> None:
        """Validate that the constant instance generator has the corrent properties."""
        assert constant_generator.num_nodes == 20
        assert constant_generator.max_length == 2

    def test_constant_generator__call(
        self, constant_generator: ConstantGenerator
    ) -> None:
        """Validate that the constant instance generator's call function is jit-table
        and compiles only once. Also check that giving two different keys results in
        two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(constant_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)


class TestUniformGenerator:
    @pytest.fixture
    def uniform_generator(self) -> UniformGenerator:
        return UniformGenerator(
            num_nodes=20,
            max_length=2,
        )

    def test_uniform_generator__properties(
        self, uniform_generator: UniformGenerator
    ) -> None:
        """Validate that the random instance generator has the corrent properties."""
        assert uniform_generator.num_nodes == 20
        assert uniform_generator.max_length == 2

    def test_uniform_generator__call(self, uniform_generator: UniformGenerator) -> None:
        """Validate that the random instance generator's call function is jit-table
        and compiles only once. Also check that giving two different keys results in
        two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(uniform_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)


class TestProportionalGenerator:
    @pytest.fixture
    def proportional_generator(self) -> ProportionalGenerator:
        return ProportionalGenerator(
            num_nodes=20,
            max_length=2,
        )

    def test_proportional_generator__properties(
        self,
        proportional_generator: ProportionalGenerator,
    ) -> None:
        """Validate that the proportional instance generator has the corrent properties."""
        assert proportional_generator.num_nodes == 20
        assert proportional_generator.max_length == 2

    def test_proportional_generator__call(
        self,
        proportional_generator: ProportionalGenerator,
    ) -> None:
        """Validate that the proportional instance generator's call function is
        jit-able and compiles only once. Also check that giving two different keys
        results in two different instances.
        """
        chex.clear_trace_counter()
        call_fn = jax.jit(chex.assert_max_traces(proportional_generator.__call__, n=1))
        state1 = call_fn(key=jax.random.PRNGKey(1))
        assert isinstance(state1, State)

        state2 = call_fn(key=jax.random.PRNGKey(2))
        assert_trees_are_different(state1, state2)
