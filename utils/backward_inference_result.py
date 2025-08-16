from dataclasses import dataclass


@dataclass
class BackwardInferenceSampleResult:
    # noinspection GrazieInspection
    """
        Represents the result of a backward inference operation.

        Attributes:
            original_prefix_tokens (list[int]): Tokens of the original prefix (X).
            original_prefix_text (str): Text of the original prefix (X).
            predicted_prefix_tokens (list[int]): Tokens of the predicted prefix (X').
            predicted_prefix_text (str): Text of the predicted prefix (X').
            suffix_tokens (list[int]): Tokens of the suffix (Y).
            suffix_text (str): Text of the suffix (Y).
            bigram_tokens (list[int] | None): Tokens of the bigram, if available (X' bigram-only).
            bigram_text (str | None): Text representation of the bigram, if available (X' bigram-only).
        """
    original_prefix_tokens: list[int]
    original_prefix_text: str
    predicted_prefix_tokens: list[int]
    predicted_prefix_text: str
    suffix_tokens: list[int]
    suffix_text: str
    bigram_tokens: list[int] | None
    bigram_text: str | None

    ilm_metrics: dict | None = None  # ILM metrics obtained later in the process (inverse_lm_stats)
    bigram_metrics: dict | None = None  # Bigram metrics obtained later in the process, if we have a bigram text

    @staticmethod
    def from_dict(data: dict) -> 'BackwardInferenceSampleResult':
        """
        Create a BackwardInferenceSampleResult from a dictionary.

        Args:
            data (dict): A dictionary containing the result data.

        Returns:
            BackwardInferenceSampleResult: An instance of the result.
        """
        return BackwardInferenceSampleResult(
            original_prefix_tokens=data['original_prefix_tokens'],
            original_prefix_text=data['original_prefix_text'],
            predicted_prefix_tokens=data['predicted_prefix_tokens'],
            predicted_prefix_text=data['predicted_prefix_text'],
            suffix_tokens=data['suffix_tokens'],
            suffix_text=data['suffix_text'],
            bigram_tokens=data.get('bigram_tokens'),
            bigram_text=data.get('bigram_text'),
            ilm_metrics=data.get('ilm_metrics'),
            bigram_metrics=data.get('bigram_metrics'),
        )

    def to_dict(self) -> dict:
        """
        Convert the result to a dictionary.

        Returns:
            dict: A dictionary representation of the result.
        """
        return {
            'original_prefix_tokens': self.original_prefix_tokens,
            'original_prefix_text': self.original_prefix_text,
            'predicted_prefix_tokens': self.predicted_prefix_tokens,
            'predicted_prefix_text': self.predicted_prefix_text,
            'suffix_tokens': self.suffix_tokens,
            'suffix_text': self.suffix_text,
            'bigram_tokens': self.bigram_tokens,
            'bigram_text': self.bigram_text,
            'ilm_metrics': self.ilm_metrics,
            'bigram_metrics': self.bigram_metrics,
        }


@dataclass
class BackwardInferenceResult:
    """
    Represents the result of a backward inference operation.

    Attributes:
        samples (list[BackwardInferenceSampleResult]): List of sample results.
        ckpt_file (str): Path to the checkpoint file used for inference.
        prefix_len (int): Length of the prefix used in the inference.
        use_init (str): The initialization method used for the inference.
        baseline_ckpt_file (str | None): Path to the baseline checkpoint file, if any.
        k_samples (int): Number of samples used in the inference.
        skip_prefix_tokens (int): Number of prefix tokens to skip before inverting.
        beam_size (int): Size of the beam used in the inference.
    """
    samples: list[BackwardInferenceSampleResult]
    ckpt_file: str
    prefix_len: int
    use_init: str
    baseline_ckpt_file: str | None
    k_samples: int
    skip_prefix_tokens: int
    beam_size: int

    @staticmethod
    def from_dict(data: dict) -> 'BackwardInferenceResult':
        """
        Create a BackwardInferenceResult from a dictionary.

        Args:
            data (dict): A dictionary containing the result data.

        Returns:
            BackwardInferenceResult: An instance of the result.
        """
        return BackwardInferenceResult(
            samples=[BackwardInferenceSampleResult.from_dict(sample) for sample in data['samples']],
            ckpt_file=data['ckpt_file'],
            prefix_len=data['prefix_len'],
            use_init=data['use_init'],
            baseline_ckpt_file=data.get('baseline_ckpt_file'),
            k_samples=data['k_samples'],
            skip_prefix_tokens=data['skip_prefix_tokens'],
            beam_size=data['beam_size'],
        )

    def to_dict(self) -> dict:
        """
        Convert the result to a dictionary.

        Returns:
            dict: A dictionary representation of the result.
        """
        return {
            'samples': [sample.to_dict() for sample in self.samples],
            'ckpt_file': self.ckpt_file,
            'prefix_len': self.prefix_len,
            'use_init': self.use_init,
            'baseline_ckpt_file': self.baseline_ckpt_file,
            'k_samples': self.k_samples,
            'skip_prefix_tokens': self.skip_prefix_tokens,
            'beam_size': self.beam_size,
        }
