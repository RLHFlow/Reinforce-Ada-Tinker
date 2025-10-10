import math
from functools import partial
from typing import Literal, Sequence

import chz
from datasets import load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class MathEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "math_verify",
        timeout: float = 1.0,
        format_coef: float = 0.0,
    ):
        super().__init__(renderer, convo_prefix, format_coef)
        self.problem = problem
        self.answer = answer
        self.grader = grader
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        return " Let's think step by step and output the final answer within \\boxed{}."

    def get_question(self) -> str:
        return self.problem

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return safe_grade(answer, self.answer, self.grader, self.timeout)

    @staticmethod
    def standard_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
            },
        ]


def safe_grade(given_answer: str, ground_truth: str, grader: str = "sympy", timeout: float = 1.0):
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out


class ReinforceAdaDataset(RLDataset):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        self.ds = load_dataset(dataset_name, split="train")
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.question_suffix = (
            " Let's think step by step and output the final answer within \\boxed{}."
        )
        self.current_seed = 0

    def set_epoch(self, seed: int) -> None:
        """Shuffle the dataset for the new epoch."""
        self.current_seed = seed
        self.ds = self.ds.shuffle(seed=seed)
        logger.info(f"Shuffled dataset with seed={seed}")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("problem", "")
        answer = x.get("gt", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem + self.question_suffix,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
            ),
            num_envs=group_size,
            dataset_name=self.dataset_name,
        )


@chz.chz
class ReinforceAdaDatasetBuilder(RLDatasetBuilder):
    dataset_name: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"

    async def __call__(self) -> tuple[ReinforceAdaDataset, None]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return ReinforceAdaDataset(
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
        ), None
