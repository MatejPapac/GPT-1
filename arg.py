import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='This is demonstration program')

    #here we add an argument to the parser, specifiying the expcected type, a help message etc.
    parser.add_argument('-bs', type=str, required=True, help = 'Please provide an a batch_size')

    return parser.parse_args()

def main():
    args = parse_args()

    #now we can use the argument value in our program
    print(f'Batch size:{args.bs}')

if __name__ == '__main__':
    main()