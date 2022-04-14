import numpy as np

__all__ = ["raise_", "string_to_boolean", "add_stats", "get_fig_size"]


def raise_(ex):
    """Raise an exception as a statement.

    This is a general purpose raiser for cases such as a lambda function.

    Parameters
    ----------
    ex: exception
        Python built-in exception to raise.
    """
    raise ex


def string_to_boolean(variables):
    """Convert True/False strings to booleans.

    Useful in case a specific use of the CLI doesn't allow to read booleans as booleans.

    Parameters
    ----------
    variables: list of str
        Variables to check.
    """

    def check_str(x):
        return (
            x
            if type(x) == bool
            else True
            if x == "True"
            else False
            if x == "False"
            else raise_(ValueError(f"{x} is not a valid boolean."))
        )

    return list(map(check_str, variables))


def add_stats(data, ax, x=0.70, y=0.85, fontsize=10):
    """Add a textbox containing statistical information."""
    mu = data.mean()
    median = np.median(data)
    sigma = data.std()
    textstr = "\n".join(
        (
            r"$\mu=%.2f$" % (mu,),
            r"$\mathrm{median}=%.2f$" % (median,),
            r"$\sigma=%.2f$" % (sigma,),
        )
    )

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        x,
        y,
        textstr,
        transform=ax.transAxes,
        fontsize=fontsize,
        horizontalalignment="left",
        verticalalignment="center",
        bbox=props,
    )


def get_fig_size(ratio=None, scale=None):
    """Get size of figure given a ratio and a scale.

    Parameters
    ----------
    ratio: float
        Something like 16:9 or 4:3
    scale: float
        A multiplicative factor to ccale the original figure keeping its ratio"""
    ratio = 4 / 3.0 if ratio is None else ratio
    scale = 1.0 if scale is None else scale
    height = 5
    width = height * ratio
    return (width * scale, height * scale)
