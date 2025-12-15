import json
import argparse
from tqdm import tqdm
from hierachy_diagnosis import hierachy_diagnosis
from utils import check_api, extract_option, count_token_usage
from dataset import DataLoader
from agents import Agent
from datasets import load_dataset
from logger_util import init_global_logger, cleanup_global_logger, log_function_calls

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pathvqa')
    parser.add_argument('--unify_model', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--checkapi', type=bool, default=False)
    
    parser.add_argument('--log_dir', type=str, default='logs', help='logger direction')
    parser.add_argument('--disable_logging', action='store_true', help='ban logger')
    
    return parser.parse_args()

@log_function_calls
def initialize_dataset(args):
    print(f"Loading Dataset: {args.dataset}")
    case_set = DataLoader(dataset=args.dataset, num_samples=args.num_samples)
    test_case, gt_options = case_set.build_all_data()
    print(f"Dataset Loaded Successfully, Totally {len(test_case)} Samples")
    return test_case, gt_options

@log_function_calls
def process_single_case(no, dataset, current_case, gt_options):
    print(f"\n[No.{no+1}]")
    print('[QUESTION]\n', current_case['question'])

    final_option, stage_end, token_usage = hierachy_diagnosis(dataset, current_case)

    print('\nCorrect Answer:', gt_options[no].strip().upper())
    print('Predicted Answer:', final_option.strip().upper())
    
    is_correct = False
    
    if gt_options[no].strip().upper() == extract_option(final_option):
        is_correct = True
    
    return is_correct, stage_end, token_usage

def print_progress(correct, tested, stats, token_usage_stats):
    accuracy = correct / tested * 100 if tested > 0 else 0
    print(f"\n\n# Current Process: {correct}/{tested} correct ({accuracy:.1f}%) # \nLEVEL-1: {stats['level-1_correct']}/{stats['level-1']}\nLEVEL-2: {stats['level-2_correct']}/{stats['level-2']}\nLEVEL-3: {stats['level-3_correct']}/{stats['level-3']}\n2->3: {stats['2->3_correct']}/{stats['2->3']}")
    print(f"Avg Prompt Tokens Per Case: {(token_usage_stats['prompt_tokens']/((tested+1e-10)*1000)):.2f}K")
    print(f"Avg Completion Tokens Per Case: {(token_usage_stats['completion_tokens']/((tested+1e-10)*1000)):.2f}K")

def main():
    args = parse_arguments()
    
    if not args.disable_logging:
        log_prefix = f"experiment_{args.dataset}"
        if args.unify_model:
            log_prefix += f"_{args.unify_model}"
        
        logger = init_global_logger(log_dir=args.log_dir, log_prefix=log_prefix)
        print("=" * 80)
        print("Experiment Starts")
        print(f"Time: {logger.log_path.name.split('_')[-1].replace('.log', '')}")
        print(f"Configs: {vars(args)}")
        print("=" * 80)
    
    try:
        if args.checkapi:
            check_api()
        
        test_case, gt_options = initialize_dataset(args)    

        stats = {
            'correct': 0,
            'tested': 0,
            'level-1': 0,
            'level-2': 0,
            'level-3': 0,
            '2->3': 0,
            'level-1_correct': 0,
            'level-2_correct': 0,
            'level-3_correct': 0,
            '2->3_correct': 0,
        }
        token_usage_stats = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

        dataset_name = args.dataset
        
        for no, current_case in enumerate(tqdm(test_case, desc="processing")):
            print_progress(stats['correct'], stats['tested'], stats, token_usage_stats)
            stats['tested'] += 1
            
            try:
                is_correct, stage_end, token_usage = process_single_case(no, dataset_name, current_case, gt_options)
                token_usage_stats = count_token_usage(token_usage_stats, token_usage)
                stats[stage_end] += 1
                if is_correct:
                    stats['correct'] += 1
                    stats[stage_end+'_correct'] += 1
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                print('[!!Error!!] Something went wrong. Continue. Samples -1.')
                stats['tested'] -= 1
                continue
                
            print('\n' + '=' * 130 + '\n')
        
        final_accuracy = stats['correct'] / stats['tested'] * 100 if stats['tested'] > 0 else 0
        print("\n" + "=" * 80)
        print("Experiment Finished!")
        print(f"Final Accuracy: {final_accuracy:.2f}% - {stats['correct']}/{stats['tested']} correct.")
        print("=" * 80)
        
        results = {
            'dataset': args.dataset,
            'model': args.unify_model,
            'total_samples': stats['tested'],
            'correct_samples': stats['correct'],
            'accuracy': final_accuracy,
            'timestamp': logger.log_path.name if not args.disable_logging else None
        }
        
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        output_filename = f"{args.dataset}_{args.unify_model}_results.json"
        output_path = output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"💾 Logger Saved: {output_path}")
        
    except KeyboardInterrupt:
        print("\n user interrupt")
    except Exception as e:
        print(f" Error: {e}")
    finally:
        if not args.disable_logging:
            cleanup_global_logger()
            print("📝 Logger Saved and Closed")

if __name__ == "__main__":
    main()