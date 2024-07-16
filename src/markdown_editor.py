import argparse
import os
import glob


def list_markdown_files(directory):
    """List all markdown files in the specified directory."""
    md_files = glob.glob(os.path.join(directory, '*.md'))
    return md_files


def edit_markdown(file_path, replace_tuple):
    """Edit the specified markdown file.

    Args:
        file_path (str): Path to the markdown file.
        replace_tuple (tuple): A tuple containing the text to find and the text to replace it with.
    """
    with open(file_path, 'r+') as f:
        contents = f.read()
        contents = contents.replace(replace_tuple[0], replace_tuple[1])
        f.seek(0)
        f.write(contents)
        f.truncate()


def main():
    parser = argparse.ArgumentParser(description='Edit markdown files in a given directory.')
    parser.add_argument('--dir', type=str, help='Directory containing markdown files to be edited')
    parser.add_argument('--find', type=str, help='Text to find in the markdown files')
    parser.add_argument('--replace', type=str, help='Text to replace the found text with.')
    args = parser.parse_args()

    if args.dir and args.find and args.replace:
        md_files = list_markdown_files(args.dir)
        for file in md_files:
            edit_markdown(file, (args.find, args.replace))

if __name__ == '__main__':
    main()


extend the existing markdown_editor.py with a new method that can add, remove or update the yaml data that is in the beginning of the markdown file between the two triple dashes. If this yaml block does not exists it must be created. the method shall take these arguments:
- operation: enum {'add','remove','update'}
- field: string
- value: string