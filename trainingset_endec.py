import argparse
import random
import json
from typing import List

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='STEP slot training data - Alpaca-style Training')
    parser.add_argument('--output_file', type=str,
                       default="slot_endec.json", 
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

# Create Alpaca formated .json file
def create_alpaca_dataset(
        input,
        target,
        instruction, 
        output_file: str,
) -> None:
    
    training_data = []
    
    for input_, target_, instruction_ in zip(input, target, instruction):

        # Join the list of STEP points into a single string
        input_str = '\n'.join(input_)
        source_ = instruction_ + '\n' + input_str
        
        # Create entry in Alpaca format
        entry = {
            # "instruction": instruction_,
            "source": source_,
            "target": '\n'.join(target_), 
        }
            
        training_data.append(entry)
    
    # Write to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
            #json.dump(training_data, f, indent=2)
        print(f"Successfully created {output_file} with {len(training_data)} training examples")
        
    except Exception as e:
        print(f"Error writing output file: {str(e)}")
        raise


def main(): 
    # Loop for creating the training samples
    args = parse_args()
    presentation_precision = args.precision
    calc_precision = presentation_precision - 1
    block, slot, instruction = [], [], []

    for _ in range(args.samples):
        block_points, slot_points = [], []

        width = round(random.uniform(args.min_width, args.max_width), calc_precision)
        height = round(random.uniform(0.15, 0.85) * width, calc_precision)             # 15-85% of block width
        depth = round(random.uniform(1.1, 3) * width, calc_precision)                  # 101-300% of block width
        slot_height = round(random.uniform(0.15, 0.85) * height, calc_precision)       # 15-85% of block height
        slot_width = round(random.uniform(.20, .80) * width, calc_precision)

        block_points = generate_block_points(width=width, height=height, depth=depth, precision=presentation_precision)
        slot_points = block_slot_points(width=width, height=height, depth=depth, 
                                        slot_width=slot_width, slot_height=slot_height, precision=presentation_precision)
        
        if args.mode == 'implicit':
            instruction_ = f"SLOT WIDTH={slot_width:.{presentation_precision}f} HEIGHT={slot_height:.{presentation_precision}f}"
        else:
            slot_start = (width - slot_width) / 2
            slot_end = slot_start + slot_width
            instruction_ = f"SLOT START={slot_start:.{presentation_precision}f} SLOT END={slot_end:.{presentation_precision}f} HEIGHT={slot_height:.{presentation_precision}f}"
    
        block.append(block_points)
        slot.append(slot_points)
        instruction.append(instruction_)
    
    create_alpaca_dataset(input=block, target=slot, instruction=instruction, output_file=args.output_file)

if __name__ == "__main__":
    main() 