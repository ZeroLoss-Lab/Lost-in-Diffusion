# Visualization

This repository contains visualization tools for the LLaDA project.

## Implementation Steps

### Step 1: Generate Sampling Process
Run `generate.py` to produce your own sampling process records. A sample output (`sample_process.txt`) is included for reference. You have the option to:
- Utilize the provided generate.py script
- Modify both the prompt and generation parameters

### Step 2: Generate Visualization HTML
Choose between two visualization styles:
- **Paper Style**:  
  `visualization_paper.py` produces visualizations matching the format in [our arXiv paper](https://arxiv.org/abs/2502.09992)
- **Zhihu Style**:  
  `visualization_zhihu.py` generates visualizations compatible with [Zhihu's format](https://zhuanlan.zhihu.com/p/24214732238)

The scripts will:
1. Automatically create an `html/` directory
2. Generate individual HTML files for each sampling step

*Note: The current implementation defaults to 64 sampling steps. 

### Step 3: Create PNG Sequences
Convert generated HTML files to PNG format for GIF creation. These image sequences can be used with any standard GIF generator to visualize the complete sampling process.

## Technical Notes
- Ensure Python 3.8+ environment
- Install required dependencies: `pip install html2image`
- For custom configurations, modify constants at the beginning of each script
