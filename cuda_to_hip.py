#!/usr/bin/python
""" The Python Hipify script.
##
# Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""

import argparse
import constants
import re
import shutil
import sys
import os

from functools import reduce
from cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS


def update_progress_bar(total, progress):
    """
    Displays and updates a console progress bar.
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()


def walk_over_directory(path, extensions, show_detailed):
    """ Walks over the entire directory and applies the function (func) on each file encountered.

    func (path as string): void
    """
    cur = 0
    total = sum([sum([reduce(lambda result, ext: filename.endswith(ext) or result, extensions, False) for filename in files]) for r, d, files in os.walk(path)])
    stats = {"unsupported_calls": [], "kernel_launches": []}

    for (dirpath, _dirnames, filenames) in os.walk(path):
        for filename in filenames:
            # Extract the (.hip)
            if filename.endswith(".hip"):
                hip_source = os.sep.join([dirpath, filename])
                dest_file = hip_source[0:-4]

                # Execute the preprocessor on the specified file.
                shutil.copy(hip_source, dest_file)

                # Assume (.hip) files are already preprocessed. Continue.
                continue

            if reduce(
                lambda result, ext: filename.endswith(ext) or result,
                    extensions, False):
                filepath = os.sep.join([dirpath, filename])

                # Execute the preprocessor on the specified file.
                preprocessor(filepath, stats)

                # Update the progress
                print(os.path.join(dirpath, filename))
                update_progress_bar(total, cur)

                cur += 1

    print("Finished")
    compute_stats(stats, show_detailed)


def compute_stats(stats, show_detailed):
    unsupported_calls = set(cuda_call for (cuda_call, _filepath) in stats["unsupported_calls"])

    # Print the number of unsupported calls
    print("Total number of unsupported CUDA function calls: %d" % (len(unsupported_calls)))

    # Print the list of unsupported calls
    print(", ".join(unsupported_calls))

    # Print the number of kernel launches
    print("\nTotal number of replaced kernel launches: %d" % (len(stats["kernel_launches"])))

    if show_detailed:
        print("\n".join(stats["kernel_launches"]))

        for unsupported in stats["unsupported_calls"]:
            print("Detected an unsupported function %s in file %s" % unsupported)


def processKernelLaunches(string, stats):
    """ Replace the CUDA style Kernel launches with the HIP style kernel launches."""
    def create_hip_kernel(cuda_kernel):
        kernel_name = cuda_kernel.group(1)
        kernel_template = cuda_kernel.group(2) if cuda_kernel.group(2) else ""
        kernel_launch_params = cuda_kernel.group(3)

        # Convert kernel launch params to list
        kernel_launch_params = kernel_launch_params.replace("<<<", "").replace(">>>", "").split(",")
        kernel_launch_params[0] = "dim3(%s)" % kernel_launch_params[0].strip()
        kernel_launch_params[1] = "dim3(%s)" % kernel_launch_params[1].strip()

        # Fill empty kernel params with 0s (sharedSize, stream)
        kernel_launch_params[len(kernel_launch_params):4] = ["0"] * (4 - len(kernel_launch_params))

        # Create the Hip Kernel Launch
        hip_kernel_launch = "".join("hipLaunchKernelGGL((%s%s), %s, " % (kernel_name, kernel_template, ", ".join(kernel_launch_params)))

        # Clean up syntax
        hip_kernel_launch = re.sub(' +', ' ', hip_kernel_launch)

        # Update stats
        stats["kernel_launches"].append(hip_kernel_launch)
        return hip_kernel_launch

    # Replace CUDA with HIP Kernel launch
    output_string = re.sub(r'(\w+)(<.*>)[\n|\\| ]+<<<(.*)>>>\(', create_hip_kernel, string)

    return output_string


def disable_asserts(input_string):
    """ Disables regular assert statements
    e.g. "assert(....)" -> "/*assert(....)*/"
    """
    def whitelist(input):
        return input.group(1) + "//" + input.group(2) + input.group(3)

    return re.sub(r'(^|[^a-zA-Z0-9_.\n]+)(assert)([^a-zA-Z0-9_.\n]+)', whitelist, input_string)


def preprocessor(filepath, stats):
    """ Executes the CUDA -> HIP conversion on the specified file. """
    with open(filepath, "r+") as fileobj:
        output_source = fileobj.read()

        # Perform type, method, constant replacements
        for mapping in CUDA_TO_HIP_MAPPINGS:
            for key, value in mapping.iteritems():
                # Extract relevant info
                cuda_type = key
                hip_type = value[0]
                meta_data = value[1:]

                if output_source.find(cuda_type) > -1:
                    # Check if supported
                    if constants.HIP_UNSUPPORTED in meta_data:
                        stats["unsupported_calls"].append((cuda_type, filepath))

                # Replace all occurances
                if cuda_type in output_source:
                    output_source = re.sub(
                        r'\b(%s)\b' % cuda_type,
                        lambda input: hip_type,
                        output_source)

        # Perform Kernel Launch Replacements
        output_source = processKernelLaunches(output_source, stats)

        # Disable asserts
        output_source = disable_asserts(output_source)

        # Overwrite file contents
        fileobj.seek(0)
        fileobj.write(output_source)
        fileobj.truncate()
        fileobj.flush()

        # Flush to disk
        os.fsync(fileobj)


def file_specific_replacement(filepath, search_string, replace_string):
    with open(filepath, "r+") as f:
        contents = f.read()
        contents = contents.replace(search_string, replace_string)
        f.seek(0)
        f.write(contents)
        f.truncate()
        f.flush()
        os.fsync(f)


def main():
    """Example invocation

    python hipify.py --project-directory /home/myproject/ --extensions cu cuh h cpp --output-directory /home/gains/
    """

    parser = argparse.ArgumentParser(
        description="The Python Hipify Script.")

    parser.add_argument(
        '--project-directory',
        type=str,
        default=os.getcwd(),
        help="The root of the project.",
        required=True)

    parser.add_argument(
        '--show-detailed',
        type=bool,
        default=False,
        help="Show detailed summary of the hipification process.",
        required=False)

    parser.add_argument(
        '--extensions',
        nargs='+',
        default=["cu", "cuh", "c", "cpp", "h", "in"],
        help="The extensions for files to run the Hipify script over.",
        required=False)

    parser.add_argument(
        '--output-directory',
        type=str,
        default="",
        help="The directory to store the hipified project.",
        required=False)

    args = parser.parse_args()

    # Sanity check arguments
    if not os.path.exists(args.project_directory):
        print("The project folder specified does not exist.")
        return

    # If output directory not set, provide a default output directory.
    if args.output_directory is "":
        args.project_directory = args.project_directory[0:-1] if args.project_directory.endswith("/") else args.project_directory
        args.output_directory = args.project_directory + "_amd"

    # Make sure output directory doesn't already exist.
    if os.path.exists(args.output_directory):
        print("The provided output directory already exists. Please move or delete it to prevent overwriting of content.")
        return

    # Remove periods from extensions
    args.extensions = map(lambda ext: ext[1:] if ext[0] is "." else ext, args.extensions)

    # Copy the directory
    shutil.copytree(args.project_directory, args.output_directory)

    # Start Preprocessor
    walk_over_directory(
        args.output_directory,
        extensions=["cu", "cuh", "c", "cpp", "h", "in"],
        show_detailed=args.show_detailed
        )


if __name__ == '__main__':
    main()
