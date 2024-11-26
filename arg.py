import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='This is demonstration program')

    #here we add an argument to the parser, specifiying the expcected type, a help message etc.
    parser.add_argument('-device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('-batch_size', type=int, default=48, help='Batch size for training and evaluation')
    parser.add_argument('-block_size', type=int, default=128, help='Block size for transformer model')
    parser.add_argument('-max_iters', type=int, default=350, help='Maximum iterations for training')
    parser.add_argument('-learning_rate', type=float, default=7.5e-5, help='Learning rate for optimizer')
    parser.add_argument('-eval_iters', type=int, default=50, help='Evaluation iterations during training')
    parser.add_argument('-n_embd', type=int, default=384, help='Embedding size for transformer model')
    parser.add_argument('-n_head', type=int, default=4, help='Number of attention heads in transformer')
    parser.add_argument('-n_layer', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('-dropout', type=float, default=0.4, help='Dropout value for the model')

    return parser.parse_args()

def main():
    args = parse_args()

    #now we can use the argument value in our program
    print(f'Batch size:{args.bs}')

if __name__ == '__main__':
    main()