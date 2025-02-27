import sys
import os
import argparse
import colorama
from thefuck.specific.llm import generate_command
from thefuck.logs import color, debug, failed
from thefuck import const


def handle_generate_command(args):
    """Handle command generation requests"""
    if not args:
        failed("No description provided")
        return 1
    query = ' '.join(args)
    
    # Call LLM to generate command
    command = generate_command(query)
    
    if not command:
        failed("Failed to generate command. Please try with a different description.")
        return 1
    
    bold = color(colorama.Style.BRIGHT)
    reset = color(colorama.Style.RESET_ALL)
    
    # Ask user whether to execute
    sys.stderr.write(
        (f"{bold}{command}{reset} "
         f"[{color(colorama.Fore.GREEN)}enter{reset}/"
         f"{color(colorama.Fore.RED)}ctrl+c{reset}]"))
    
    try:
        response = input()
        if response.strip() == '':
            # os.system(command)
            print(command)
    except KeyboardInterrupt:
        sys.stderr.write('\nAborted\n')
    
    return 0


def main():
    """Main entry function"""
    parser = argparse.ArgumentParser(description='Generate command from natural language description')
    parser.add_argument('query', nargs='+', help='Natural language command description')
    
    args = parser.parse_args()
    
    if not args.query:
        failed("Usage: fuckgenerate <command description>")
        return 1
    
    return handle_generate_command(args.query)


if __name__ == "__main__":
    sys.exit(main())