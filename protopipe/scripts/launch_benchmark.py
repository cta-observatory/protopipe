"""Script to launch one of the benchmark notebooks stored
in the documentation and display the result in browser."""

import argparse
import glob
import logging
from pathlib import Path
import pkg_resources
import webbrowser
import yaml
import shutil

from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
import papermill as pm
from traitlets.config import Config

from protopipe.pipeline.io import load_config


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def main():

    # create the top-level parser
    parser = argparse.ArgumentParser(
        description="""
        Launch a benchmark notebook within an analysis Jupyter Book.
        Optionally convert each notebook also into an HTML page.

        USAGE EXAMPLE:
        --------------
        >>> protopipe-BENCHMARK list
        >>> protopipe-BENCHMARK launch -n TRAINING/benchmarks_DL1b_image-cleaning --config_file benchmarks.yaml
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # create the parser for the "launch" command
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("list", help="List available benchmarks")

    # create the parser for the "launch" command
    # subparsers = parser.add_subparsers()
    parser_launch = subparsers.add_parser("launch", help="Launch a specific benchmark")
    parser_launch.add_argument(
        "--help-notebook",
        action="store_true",
        help="Print the list of available notebook parameters",
    )

    parser_launch.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Pipeline step and name of the benchmark (for a list use `protopipe-BENCHMARK -l`)",
    )

    parser_launch.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Configuration file (default: stored under analysis 'config' folder)",
    )

    parser_launch.add_argument(
        "-k",
        "--kwargs",
        nargs="*",
        action=ParseKwargs,
        help="Overwrite or specify other configuration options (e.g. --kwargs foo=bar fiz=biz)",
    )

    parser_launch.add_argument(
        "--outpath",
        type=str,
        default=None,
        help="Specificy an analysis folder (recommended: use benchmarks.yaml)",
    )

    parser_launch.add_argument(
        "--overwrite_notebook",
        action="store_true",
        help="Execute the notebook even if it overwrites the old result.",
    )

    parser_launch.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix for result and HTML files",
    )

    parser_launch.add_argument(
        "--no_export",
        action="store_true",
        help="Do not convert the result notebook to any other format.",
    )

    args = parser.parse_args()

    # Define the path containing all available benchmarks
    BENCHMARKS_PATH = Path(
        pkg_resources.resource_filename("protopipe", "benchmarks/notebooks")
    )

    if args.command == "list":
        # Select only notebooks which name starts with benchmarks*
        # This is to not list also results
        notebooks = glob.glob(str((BENCHMARKS_PATH / "**/b*.ipynb")), recursive=True)
        for notebook in notebooks:
            print(notebook.split("benchmarks/notebooks/")[1].split(".")[0])
        exit()
    elif args.command == "launch":

        # Read config and add or overwrite any additional CLI configuration options
        cfg = load_config(args.config_file)
        if args.kwargs:
            cfg.update(args.kwargs)

        if args.outpath:
            output_directory = Path(args.outpath)
        else:
            output_directory = Path(cfg["analyses_directory"])

        # Define book template path and create it (only the first time)
        BOOK_TEMPLATE = Path(
            pkg_resources.resource_filename("protopipe", "benchmarks/book_template")
        )
        outdir_book = output_directory / cfg["analysis_name"] / "book"
        Path.mkdir(outdir_book, parents=True, exist_ok=True)
        # copy recursevely the template files only if they are not already there
        if not all(
            [
                (outdir_book / f).exists()
                for f in [Path(f).name for f in glob.glob(str(BOOK_TEMPLATE / "*"))]
            ]
        ):
            # copy contents recursively
            shutil.copytree(BOOK_TEMPLATE, outdir_book, dirs_exist_ok=True)
        else:
            logging.warning("Analysis' book files already exist - not overwriting.")

        # Define path variables for the "launch" sub-command
        suffix = f"_{args.suffix}" if args.suffix else ""
        data_level = args.name.split("/")[0]
        benchmark_name = args.name.split("/")[1]
        input_notebook = Path(BENCHMARKS_PATH / data_level / f"{benchmark_name}.ipynb")
        outdir_notebook = (
            output_directory / cfg["analysis_name"] / "book" / "results" / data_level
        )
        Path.mkdir(outdir_notebook, parents=True, exist_ok=True)
        result_notebook = Path(
            outdir_notebook / f"result_{benchmark_name}{suffix}.ipynb"
        )

        # Some special cases
        if args.help_notebook:
            parameters = pm.inspect_notebook(input_notebook)
            print(yaml.dump(parameters))
            exit()
        if (not args.overwrite_notebook) and Path(result_notebook).is_file():
            logging.critical(
                "Result notebook exists. To overwrite it use '--overwrite_notebook'."
            )
            exit()

        # create output directory if necessary
        Path.mkdir(Path(result_notebook.parent), parents=True, exist_ok=True)
        cfg["output_directory"] = str(outdir_notebook)
        pm.execute_notebook(input_notebook, result_notebook, parameters=cfg)

        if args.no_export:
            logging.info("Execution completed.")
        else:

            # Use jupyter nbconvert to convert Markdown and output to HTML

            # Setup config
            c = Config()
            c.TagRemovePreprocessor.enabled = True
            c.TagRemovePreprocessor.remove_cell_tags = set(["remove_input"])

            # Configure and run out exporter
            c.HTMLExporter.preprocessors = [
                "nbconvert.preprocessors.TagRemovePreprocessor"
            ]
            c.HTMLExporter.exclude_input = True

            html_exporter = HTMLExporter(config=c)
            html_exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)

            # Configure and run exporter - returns a tuple - first element with html,
            # second with notebook metadata
            output = html_exporter.from_filename(result_notebook)

            # Write to output html file
            html_notebook = Path(
                outdir_notebook / f"result_{benchmark_name}{suffix}.html"
            )
            with open(html_notebook, "w") as f:
                f.write(output[0])

            # open HTML file
            new = 2  # open in a new tab, if possible
            url = "file://" + str(html_notebook)
            webbrowser.open(url, new=new)

            logging.info("Conversion completed.")

    else:

        logging.critical(
            "Available commands for protopipe-BENCHMARK are `list` or `launch`"
        )
        exit()


if __name__ == "__main__":
    main()
