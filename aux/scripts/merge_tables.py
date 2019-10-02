#!/usr/bin/env python

import argparse
import glob

# PyTables
try:
    import tables as tb
except ImportError:
    print("no pytables installed?")


def main():
    parser = argparse.ArgumentParser(description="Merge collection of HDF5 files")
    parser.add_argument("--indir", type=str, default="./")
    parser.add_argument("--template_file_name", type=str, default="features_event")
    parser.add_argument("--outfile", type=str)
    args = parser.parse_args()

    print("DEBUG> template_file_name={}".format(args.template_file_name))
    print("DEBUG> indir={}".format(args.indir))
    print("DEBUG> outfile={}".format(args.outfile))

    input_template = "{}/{}*.h5".format(args.indir, args.template_file_name)
    print("input_template:", input_template)

    filename_list = glob.glob(input_template)
    print("filename_list (truncated):", filename_list[0:10])

    merge_list_of_pytables(filename_list, args.outfile)


def merge_list_of_pytables(filename_list, destination):
    merged_tables = {}
    outfile = tb.open_file(destination, mode="w")

    for idx, filename in enumerate(sorted(filename_list)):

        infile = tb.open_file(filename, mode="r")
        table_name_list = [table.name for table in infile.root]  # Name of tables

        # Initialise output file
        if idx == 0:
            for name in table_name_list:
                merged_tables[name] = infile.copy_node(
                    where="/", name=name, newparent=outfile.root
                )
        else:
            for name in table_name_list:
                table_tmp = infile.get_node("/" + name)
                table_tmp.append_where(dstTable=merged_tables[name])

        infile.close()

    return merged_tables


if __name__ == "__main__":

    main()
