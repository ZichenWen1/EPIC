import sys
import argparse
from llava.train.train_TCD import train

def main():
    parser = argparse.ArgumentParser(description="TCD training script for token pruning methods")
    parser.add_argument("--method", type=str, default="dart", 
                       choices=["dart", "fastv", "random"],
                       help="Pruning method to use")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                       choices=["flash_attention_2", "sdpa", "eager"],
                       help="Attention implementation")
    
    args, remaining_args = parser.parse_known_args()
    
    # Add the selected method as the --pruning_method argument
    remaining_args.extend(["--pruning_method", args.method])
    
    # Set sys.argv for HfArgumentParser to process remaining arguments
    sys.argv = ["train_TCD.py"] + remaining_args
    
    # Select the attention implementation according to the method
    if args.method == "fastv":
        attn_impl = "sdpa"  # FastV does not support flash attention
    else:
        attn_impl = args.attn_implementation
    
    train(attn_implementation=attn_impl)

if __name__ == "__main__":
    main()
