"""Directional probability samplers used by bee movement strategies."""
import numpy as np

def _create_pdf_sampler_from_discrete(x_discrete, y_discrete):
    """
    Internal helper: builds a continuous directional PDF sampler
    from a discrete distribution using interpolation.

    Args:
        x_discrete (np.ndarray): Angles in degrees (e.g., [-180, -135, ..., 180])
        y_discrete (np.ndarray): Probability weights (non-normalized)

    Returns:
        function: A sampler function taking (rng, n) and returning a list of n angles
    """
    x_fine = np.linspace(-180, 180, 1000)
    pdf_fine = np.interp(x_fine, x_discrete, y_discrete)
    dx_pdf = x_fine[1] - x_fine[0]
    pdf_fine /= np.sum(pdf_fine * dx_pdf)  # Normalize
    cdf_pdf = np.cumsum(pdf_fine) * dx_pdf
    cdf_pdf /= cdf_pdf[-1]

    def sampler(rng, n=1):
        return [np.interp(rng.random(), cdf_pdf, x_fine) for _ in range(n)]

    return sampler


def create_biased_pdf_sampler():
    """
    Creates a directional sampler biased toward forward movement (0 degrees).
    This matches the PDF currently used in your model.py.

    Returns:
        function: biased_distribution(rng, n=1) -> list of angles
    """
    x_discrete = np.array([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    y_discrete = np.array([0.06, 0.07, 0.13, 0.17, 0.20, 0.17, 0.13, 0.07, 0.06])
    return _create_pdf_sampler_from_discrete(x_discrete, y_discrete)


def create_uniform_pdf_sampler():
    """
    Creates a uniform sampler over [-180, 180] degrees.
    Useful as a baseline or control behavior.

    Returns:
        function: uniform_distribution(rng, n=1) -> list of angles
    """
    def uniform_sampler(rng, n=1):
        return [rng.uniform(-180, 180) for _ in range(n)]
    return uniform_sampler


def get_pdf_sampler(name="biased"):
    """
    General selector for predefined directional samplers.

    Args:
        name (str): One of "biased", "uniform"

    Returns:
        function: directional sampler
    """
    if name == "biased":
        return create_biased_pdf_sampler()
    elif name == "uniform":
        return create_uniform_pdf_sampler()
    else:
        raise ValueError(f"Unknown PDF sampler: {name}")
