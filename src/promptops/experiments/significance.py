"""Prompt A/B test statistical significance calculator.

Provides chi-squared test, confidence intervals, and minimum sample
size estimation for prompt A/B experiments.

Usage:
    from promptops.experiments.significance import SignificanceCalculator

    calc = SignificanceCalculator(confidence_level=0.95)
    result = calc.test(
        control_successes=85, control_total=100,
        treatment_successes=92, treatment_total=100,
    )
    print(f"Significant: {result.is_significant}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Lift: {result.lift_percent:.1f}%")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SignificanceResult:
    """Result of a statistical significance test."""

    is_significant: bool
    p_value: float
    chi_squared: float
    confidence_level: float
    control_rate: float
    treatment_rate: float
    lift_percent: float
    lift_absolute: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    sample_size_adequate: bool
    minimum_sample_needed: int
    power: float  # Statistical power (1 - beta)
    recommendation: str


@dataclass
class SampleSizeEstimate:
    """Minimum sample size estimation result."""

    min_samples_per_variant: int
    total_samples_needed: int
    baseline_rate: float
    minimum_detectable_effect: float
    confidence_level: float
    power: float
    estimated_duration_days: Optional[float] = None


class SignificanceCalculator:
    """Statistical significance calculator for prompt A/B tests.

    Implements chi-squared test for proportions, confidence intervals
    using the Wald method, and sample size estimation using the
    normal approximation.

    Args:
        confidence_level: Target confidence level (default 0.95 = 95%).
        min_observations: Minimum observations per variant before testing.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_observations: int = 30,
    ):
        if not 0.8 <= confidence_level <= 0.99:
            raise ValueError("confidence_level must be between 0.80 and 0.99")
        self._confidence_level = confidence_level
        self._min_observations = min_observations
        self._alpha = 1.0 - confidence_level

    @property
    def confidence_level(self) -> float:
        """The configured confidence level."""
        return self._confidence_level

    @property
    def alpha(self) -> float:
        """The significance threshold (1 - confidence_level)."""
        return self._alpha

    def test(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
    ) -> SignificanceResult:
        """Run a significance test comparing control vs treatment.

        Uses the chi-squared test for two proportions.

        Args:
            control_successes: Number of successes in control group.
            control_total: Total observations in control group.
            treatment_successes: Number of successes in treatment group.
            treatment_total: Total observations in treatment group.

        Returns:
            SignificanceResult with test outcome and metrics.

        Raises:
            ValueError: If inputs are invalid.
        """
        if control_total <= 0 or treatment_total <= 0:
            raise ValueError("Total observations must be positive")
        if control_successes < 0 or treatment_successes < 0:
            raise ValueError("Successes cannot be negative")
        if control_successes > control_total or treatment_successes > treatment_total:
            raise ValueError("Successes cannot exceed total observations")

        # Calculate rates
        control_rate = control_successes / control_total
        treatment_rate = treatment_successes / treatment_total

        # Calculate lift
        lift_absolute = treatment_rate - control_rate
        lift_percent = (lift_absolute / control_rate * 100) if control_rate > 0 else 0.0

        # Chi-squared test for two proportions
        chi_squared, p_value = self._chi_squared_test(
            control_successes, control_total,
            treatment_successes, treatment_total,
        )

        # Confidence interval for the difference
        ci_lower, ci_upper = self._confidence_interval(
            control_rate, control_total,
            treatment_rate, treatment_total,
        )

        # Check sample adequacy
        sample_adequate = (
            control_total >= self._min_observations
            and treatment_total >= self._min_observations
        )

        # Minimum sample size needed
        min_sample = self._minimum_sample_size(
            control_rate, abs(lift_absolute) if lift_absolute != 0 else 0.05
        )

        # Statistical power
        power = self._calculate_power(
            control_rate, treatment_rate, control_total, treatment_total
        )

        # Determine significance
        is_significant = p_value < self._alpha and sample_adequate

        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_significant, p_value, lift_percent, power, sample_adequate,
            control_total + treatment_total, min_sample * 2,
        )

        return SignificanceResult(
            is_significant=is_significant,
            p_value=p_value,
            chi_squared=chi_squared,
            confidence_level=self._confidence_level,
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            lift_percent=lift_percent,
            lift_absolute=lift_absolute,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            sample_size_adequate=sample_adequate,
            minimum_sample_needed=min_sample * 2,
            power=power,
            recommendation=recommendation,
        )

    def estimate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.80,
        daily_traffic: Optional[int] = None,
    ) -> SampleSizeEstimate:
        """Estimate minimum sample size for a planned experiment.

        Args:
            baseline_rate: Current success rate (0.0 to 1.0).
            minimum_detectable_effect: Smallest difference to detect (absolute).
            power: Statistical power (default 0.80 = 80%).
            daily_traffic: Optional daily observations for duration estimate.

        Returns:
            SampleSizeEstimate with sample size and optional duration.
        """
        if not 0.0 < baseline_rate < 1.0:
            raise ValueError("baseline_rate must be between 0 and 1")
        if minimum_detectable_effect <= 0:
            raise ValueError("minimum_detectable_effect must be positive")
        if not 0.5 <= power <= 0.99:
            raise ValueError("power must be between 0.50 and 0.99")

        n = self._minimum_sample_size(baseline_rate, minimum_detectable_effect, power)
        total = n * 2  # Two variants

        duration = None
        if daily_traffic and daily_traffic > 0:
            duration = total / daily_traffic

        return SampleSizeEstimate(
            min_samples_per_variant=n,
            total_samples_needed=total,
            baseline_rate=baseline_rate,
            minimum_detectable_effect=minimum_detectable_effect,
            confidence_level=self._confidence_level,
            power=power,
            estimated_duration_days=duration,
        )

    def _chi_squared_test(
        self,
        s1: int, n1: int,
        s2: int, n2: int,
    ) -> Tuple[float, float]:
        """Perform chi-squared test for two proportions.

        Returns (chi_squared_statistic, p_value).
        """
        # Pooled proportion
        p_pooled = (s1 + s2) / (n1 + n2)

        if p_pooled == 0 or p_pooled == 1:
            return 0.0, 1.0

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        if se == 0:
            return 0.0, 1.0

        # Z-statistic
        p1 = s1 / n1
        p2 = s2 / n2
        z = (p2 - p1) / se

        # Chi-squared = z^2
        chi_sq = z * z

        # P-value from chi-squared distribution (1 df)
        # Using the survival function approximation
        p_value = self._chi_squared_survival(chi_sq, df=1)

        return chi_sq, p_value

    def _confidence_interval(
        self,
        p1: float, n1: int,
        p2: float, n2: int,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference in proportions (Wald method)."""
        diff = p2 - p1

        # Standard error of the difference
        se = math.sqrt(
            (p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)
        ) if n1 > 0 and n2 > 0 else 0.0

        z = self._z_score(self._confidence_level)

        lower = diff - z * se
        upper = diff + z * se

        return lower, upper

    def _minimum_sample_size(
        self,
        baseline_rate: float,
        effect_size: float,
        power: float = 0.80,
    ) -> int:
        """Calculate minimum sample size per variant.

        Uses the formula for comparing two proportions:
        n = (z_alpha/2 + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p2-p1)^2
        """
        if effect_size == 0:
            return self._min_observations

        p1 = baseline_rate
        p2 = baseline_rate + effect_size

        # Clamp p2 to valid range
        p2 = max(0.01, min(0.99, p2))

        z_alpha = self._z_score(self._confidence_level)
        z_beta = self._z_score(power)

        numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
        denominator = (p2 - p1) ** 2

        if denominator == 0:
            return self._min_observations

        n = math.ceil(numerator / denominator)
        return max(n, self._min_observations)

    def _calculate_power(
        self,
        p1: float, p2: float,
        n1: int, n2: int,
    ) -> float:
        """Calculate the statistical power of the test."""
        if p1 == p2 or n1 == 0 or n2 == 0:
            return 0.0

        # Effect size
        diff = abs(p2 - p1)

        # Standard error under alternative hypothesis
        se = math.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))

        if se == 0:
            return 1.0

        z_alpha = self._z_score(self._confidence_level)
        z_observed = diff / se

        # Power = P(reject H0 | H1 true) ≈ Φ(z_observed - z_alpha/2)
        power_z = z_observed - z_alpha
        power = self._normal_cdf(power_z)

        return max(0.0, min(1.0, power))

    def _z_score(self, confidence: float) -> float:
        """Get z-score for a given confidence level (two-tailed)."""
        # Common z-scores
        z_table = {
            0.80: 1.282,
            0.85: 1.440,
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576,
        }

        # Find closest
        closest = min(z_table.keys(), key=lambda k: abs(k - confidence))
        if abs(closest - confidence) < 0.005:
            return z_table[closest]

        # Linear interpolation for intermediate values
        keys = sorted(z_table.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= confidence <= keys[i + 1]:
                t = (confidence - keys[i]) / (keys[i + 1] - keys[i])
                return z_table[keys[i]] + t * (z_table[keys[i + 1]] - z_table[keys[i]])

        return 1.960  # Default to 95%

    def _normal_cdf(self, x: float) -> float:
        """Approximate the standard normal CDF using the error function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    def _chi_squared_survival(self, x: float, df: int = 1) -> float:
        """Approximate p-value from chi-squared distribution.

        Uses the Wilson-Hilferty approximation for df=1.
        """
        if x <= 0:
            return 1.0

        # For df=1, chi-squared is just the square of a standard normal
        # P(chi^2 > x) = 2 * P(Z > sqrt(x)) = 2 * (1 - Phi(sqrt(x)))
        z = math.sqrt(x)
        p = 2.0 * (1.0 - self._normal_cdf(z))
        return max(0.0, min(1.0, p))

    def _generate_recommendation(
        self,
        is_significant: bool,
        p_value: float,
        lift_percent: float,
        power: float,
        sample_adequate: bool,
        current_samples: int,
        needed_samples: int,
    ) -> str:
        """Generate a human-readable recommendation."""
        if not sample_adequate:
            return (
                f"Insufficient data. Need {needed_samples} total samples, "
                f"have {current_samples}. Continue the experiment."
            )

        if is_significant and lift_percent > 0:
            return (
                f"Treatment is significantly better (+{lift_percent:.1f}%, "
                f"p={p_value:.4f}). Recommend promoting treatment to production."
            )
        elif is_significant and lift_percent < 0:
            return (
                f"Treatment is significantly worse ({lift_percent:.1f}%, "
                f"p={p_value:.4f}). Recommend keeping control."
            )
        elif power < 0.5:
            return (
                f"Not significant (p={p_value:.4f}) but low power ({power:.0%}). "
                f"Collect more data — need ~{needed_samples} total samples."
            )
        else:
            return (
                f"No significant difference detected (p={p_value:.4f}). "
                f"Consider keeping the simpler/cheaper variant."
            )
