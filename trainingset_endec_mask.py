import argparse
import random
import json
from typing import List, Tuple

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='STEP slot training data - Masked Pre-training for Encoder-Decoder')
    parser.add_argument('--output_file', type=str,
                       default="masked_step_pretrain.json", 
                       help='file for training data')
    parser.add_argument('--mode', type=str,
                        choices=['implicit', 'explicit'],
                        help='implicit: slot is defined by width; explicit: slot is defined by start/end', 
                        default='implicit')
    parser.add_argument('--samples', type=int, 
                        default=1000, 
                        help='number of random samples generated')
    parser.add_argument('--min_width', type=int, default=1)
    parser.add_argument('--max_width', type=int, default=1000)
    parser.add_argument('--precision', type=int, default=2)
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                       help='ratio of lines to mask (0.0 to 1.0)')
    parser.add_argument('--mask_token', type=str, default='<MASK>',
                       help='token to use for masking')
    parser.add_argument('--unmasked_token', type=str, default='<UNMASKED>',
                       help='token to indicate unmasked lines in answers')
    parser.add_argument('--min_mask_lines', type=int, default=1,
                       help='minimum number of lines to mask')
    parser.add_argument('--max_mask_lines', type=int, default=5,
                       help='maximum number of lines to mask')
    return parser.parse_args()

# Input block function
def generate_block_points(width, height, depth, precision):
    """
    Generate STEP file CARTESIAN_POINT elements for a block's corners.
    
    Args:
        width (float): Block width (A dimension)
        height (float): Block height (B dimension)
        depth (float): Block depth (C dimension)
        
    Returns:
        list: List of STEP file CARTESIAN_POINT strings
    """
    # Define the 8 corners of the block
    corners = [
        (0, 0, 0),                          # Point 1 - origin
        (width, 0, 0),                      # Point 2 - width
        (width, height, 0),                 # Point 3 - width + height
        (0, height, 0),                     # Point 4 - height
        (0, 0, depth),                      # Point 5 - depth
        (width, 0, depth),                  # Point 6 - width + depth
        (width, height, depth),             # Point 7 - width + height + depth
        (0, height, depth)                  # Point 8 - height + depth
    ]
    
    # Generate STEP file format strings
    step_points = []
    for i, (x, y, z) in enumerate(corners, 1):
        point = f"#{i} = CARTESIAN_POINT('NONE',({x:.{precision}f}, {y:.{precision}f}, {z:.{precision}f}));"
        step_points.append(point)
        
    return step_points

# target block function; adds slot of specified width and height
def block_slot_points(width, height, depth, slot_width, slot_height, precision):

    slot_start = (width - slot_width) / 2
    slot_end = slot_start + slot_width

    corners = [
        (0, 0, 0),                          # Point 1 - origin
        (width, 0, 0),                      # Point 2 - width
        (width, height, 0),                 # Point 3 - width + height
        (0, height, 0),                     # Point 4 - height
        (slot_start, 0, 0),                  # Slot points
        (slot_start, slot_height, 0),        # Slot points
        (slot_end, slot_height, 0),        # Slot points
        (slot_end, 0, 0),                  # Slot points
        (0, 0, depth),                      # Point 5 - depth
        (width, 0, depth),                  # Point 6 - width + depth
        (width, height, depth),             # Point 7 - width + height + depth
        (0, height, depth),                 # Point 8 - height + depth
        (slot_start, 0, depth),              # Slot points
        (slot_start, slot_height, depth),    # Slot points
        (slot_end, slot_height, depth),    # Slot points
        (slot_end, 0, depth),              # Slot points
    ]
    
    # Generate STEP file format strings
    step_points = []
    for i, (x, y, z) in enumerate(corners, 1):
        point = f"#{i} = CARTESIAN_POINT('NONE',({x:.{precision}f}, {y:.{precision}f}, {z:.{precision}f}));"
        step_points.append(point)
        
    return step_points

def mask_step_code(step_lines: List[str], mask_ratio: float, mask_token: str, 
                   unmasked_token: str, min_mask_lines: int, max_mask_lines: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Mask random lines in STEP code for pre-training.
    
    Args:
        step_lines: List of STEP code lines
        mask_ratio: Ratio of lines to potentially mask
        mask_token: Token to use for masking
        unmasked_token: Token to indicate unmasked lines in answers
        min_mask_lines: Minimum number of lines to mask
        max_mask_lines: Maximum number of lines to mask
        
    Returns:
        Tuple of (masked_lines, masked_original_lines, answer_with_tokens)
    """
    if len(step_lines) == 0:
        return [], [], []
    
    # Determine number of lines to mask
    num_lines = len(step_lines)
    max_possible_masks = min(max_mask_lines, num_lines)
    min_possible_masks = min(min_mask_lines, num_lines)
    
    # Calculate target number of masks based on ratio
    target_masks = max(min_possible_masks, 
                      min(max_possible_masks, 
                          int(num_lines * mask_ratio)))
    
    # Randomly select lines to mask
    lines_to_mask = random.sample(range(num_lines), target_masks)
    
    # Create masked version
    masked_lines = step_lines.copy()
    masked_original = []
    answer_with_tokens = []
    
    for line_idx in lines_to_mask:
        masked_original.append(step_lines[line_idx])
        masked_lines[line_idx] = mask_token
    
    # Create answer with tokens indicating masked vs unmasked lines
    for i, line in enumerate(step_lines):
        if i in lines_to_mask:
            # This line was masked, so include the original content
            answer_with_tokens.append(line)
        else:
            # This line was unmasked, so include the unmasked token
            answer_with_tokens.append(unmasked_token)
    
    return masked_lines, masked_original, answer_with_tokens

# Create masked pre-training dataset
def create_masked_pretrain_dataset(
        masked_inputs,
        targets,
        masked_answers,
        output_file: str,
) -> None:
    
    training_data = []
    
    for masked_input, target, masked_answer in zip(masked_inputs, targets, masked_answers):

        # Join the list of STEP points into a single string
        masked_input_str = '\n'.join(masked_input)
        target_str = '\n'.join(target)
        masked_answer_str = '\n'.join(masked_answer)
        
        # Create entry for pre-training
        entry = {
            "source": masked_input_str,
            "target": masked_answer_str, # target_str,
            "answers": target_str, #masked_answer_str,
        }
            
        training_data.append(entry)
    
    # Write to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully created {output_file} with {len(training_data)} pre-training examples")
        
    except Exception as e:
        print(f"Error writing output file: {str(e)}")
        raise


def main(): 
    # Loop for creating the training samples
    args = parse_args()
    presentation_precision = args.precision
    calc_precision = presentation_precision - 1
    masked_inputs, targets, masked_answers = [], [], []

    for _ in range(args.samples):
        # Generate block dimensions
        width = round(random.uniform(args.min_width, args.max_width), calc_precision)
        height = round(random.uniform(0.15, 0.85) * width, calc_precision)             # 15-85% of block width
        depth = round(random.uniform(1.1, 3) * width, calc_precision)                  # 101-300% of block width
        slot_height = round(random.uniform(0.15, 0.85) * height, calc_precision)       # 15-85% of block height
        slot_width = round(random.uniform(.20, .80) * width, calc_precision)

        # Generate complete STEP code (target)
        complete_step = block_slot_points(width=width, height=height, depth=depth, 
                                        slot_width=slot_width, slot_height=slot_height, 
                                        precision=presentation_precision)
        
        # Create masked version for input
        masked_step, masked_original, answer_with_tokens = mask_step_code(
            step_lines=complete_step,
            mask_ratio=args.mask_ratio,
            mask_token=args.mask_token,
            unmasked_token=args.unmasked_token,
            min_mask_lines=args.min_mask_lines,
            max_mask_lines=args.max_mask_lines
        )
        
        masked_inputs.append(masked_step)
        targets.append(complete_step)
        masked_answers.append(answer_with_tokens)
    
    create_masked_pretrain_dataset(
        masked_inputs=masked_inputs, 
        targets=targets, 
        masked_answers=masked_answers, 
        output_file=args.output_file
    )

if __name__ == "__main__":
    main() 