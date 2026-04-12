import re
from typing import List, Dict
import os

def parse_generation_history(file_path: str) -> Dict[int, List[str]]:
    """Improved parser that handles math symbols and spaces correctly"""
    history = {}
    token_pattern = re.compile(r"\*([^&]*)&?")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                step_part, content_part = line.split(',', 1)
                step = int(step_part.strip())
            except ValueError:
                continue
            
            tokens = []
            for match in token_pattern.finditer(content_part):
                raw_token = match.group(1).strip()
                
                if raw_token == "":
                    tokens.append(" ")
                elif raw_token == "*":
                    tokens.append("*")
                else:
                    tokens.append(raw_token)

            while len(tokens) < 64:
                tokens.append(" ")
            
            if len(tokens) > 64:
                print(f"Truncating extra tokens: Step {step} ({len(tokens)} tokens)")
                tokens = tokens[:64]
            elif len(tokens) < 64:
                print(f"Padding missing tokens: Step {step} ({len(tokens)} tokens)")
                tokens += [" "] * (64 - len(tokens))
            
            history[step] = tokens
    
    return history

def track_token_positions(history: Dict[int, List[str]]) -> List[int]:
    """Track the first generation step for each token"""
    num_positions = 64
    steps_to_unmask = [-1] * num_positions

    for step in sorted(history.keys()):
        tokens = history[step]
        for idx in range(num_positions):
            if idx >= len(tokens):
                continue
                
            token = tokens[idx]
            if steps_to_unmask[idx] == -1 and token != '<|mdm_mask|>':
                steps_to_unmask[idx] = step
    
    return steps_to_unmask

def generate_background_color(step: int, max_step: int) -> str:
    """Generate gradient color"""
    color_stops = [
        (240, 248, 255), (209, 226, 241), (176, 202, 224), (143, 179, 207),
        (110, 156, 191), (77, 133, 175), (44, 110, 159), (12, 55, 112)
    ]
    
    color_index = min(step * 6 // max_step, 6)
    ratio = (step % 2) / 2
    
    start = color_stops[color_index]
    end = color_stops[min(color_index + 1, 7)]
    
    r = int(start[0] + (end[0] - start[0]) * ratio)
    g = int(start[1] + (end[1] - start[1]) * ratio)
    b = int(start[2] + (end[2] - start[2]) * ratio)
    
    return f"#{r:02x}{g:02x}{b:02x}"

def generate_step_visualization(current_step: int, current_tokens: List[str], 
                               token_steps: List[int], max_step: int) -> str:
    """Generate visualization for specific step"""
    html = []
    
    for idx, token in enumerate(current_tokens):
        style = [
            "color: #000000",
            "padding: 6px 8px",
            "margin: 3px",
            "border-radius: 6px",
            "display: inline-block",
            "font-weight: 600",
            "font-size: 16px",
            "font-family: 'Segoe UI', sans-serif",
            "box-shadow: 0 3px 6px rgba(12,55,112,0.15)",
            "transition: all 0.2s ease",
            "position: relative",
            "width: 120px",
            "min-width: 120px",
            "text-align: center",
            "white-space: nowrap",
            "overflow: hidden",
            "text-overflow: ellipsis",
            "box-sizing: border-box"
        ]

        if token == '<|mdm_mask|>':
            style.extend([
                "background: #f8fafc",
                "border: 2px solid #ffffff",
                "font-weight: 800",
                "text-transform: uppercase",
                "padding: 4px 6px"
            ])
            display_text = "Mask"
        else:
            bg_color = generate_background_color(token_steps[idx], max_step)
            style.append(f"background-color: {bg_color}")
            display_text = token if token != " " else "␣"
            
        html.append(f'<span style="{"; ".join(style)}">{display_text}</span>')
    
    return '\n'.join(html)

def main(target_step: int = 64):
    """Main function supporting target step specification"""
    file_path = "sample_process.txt"
    final_step = 64
    
    history = parse_generation_history(file_path)
    if target_step not in history:
        raise ValueError(f"Invalid target step: {target_step}")
    
    token_steps = track_token_positions(history)
    current_tokens = history[target_step]
    
    html_content = generate_step_visualization(
        target_step, current_tokens, token_steps, final_step
    )
    
    example_steps = [0, 16, 32, 48, 64]
    example_colors = [generate_background_color(s, final_step) for s in example_steps]
    legend_html = ''.join(
        f'<div style="background-color: {color}; color: black;">Step {s}</div>'
        for s, color in zip(example_steps, example_colors)
    )
    
    target_dir = "html/sample_process_paper"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    with open(f"{target_dir}/visualization_step_{target_step}.html", "w", encoding="utf-8") as f:
        f.write(f"""<html>
<head>
    <title>Step {target_step} Visualization</title>
    <style>
        body {{ 
            padding: 40px;
            background: #f8fafc;
            font-family: 'Segoe UI', sans-serif;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            margin: 20px 0;
        }}
        .legend div {{
            padding: 10px;
            border-radius: 5px;
            color: white;
            min-width: 80px;
            text-align: center;
        }}
        .token:hover {{
            transform: translateY(-2px);
        }}
    </style>
</head>
<body>
    <div style="max-width: 1000px; margin: auto;">
        <h2>Generation Step {target_step}</h2>
        <div>{html_content}</div>
        
        <h3>Color Legend</h3>
        <div class="legend">{legend_html}</div>
    </div>
</body>
</html>""")

if __name__ == "__main__":
    for step in range(1, 65):
        main(target_step=step)