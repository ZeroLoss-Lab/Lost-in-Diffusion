from html2image import Html2Image
import os

# Define the types array to process
types = ['zhihu', 'paper']  # Add all types you need to process here

# Initialize Html2Image object
hti = Html2Image()
hti.browser.use_new_headless = None  # Keep default settings

for type_txt in types:
    # Ensure png directory exists
    output_dir = os.path.join('png', f"sample_process_{type_txt}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set output path for current type
    hti.output_path = output_dir
    
    # Loop to generate screenshots
    for i in range(1, 65):
        # Get HTML file path
        html_path = os.path.join('html', f"sample_process_{type_txt}", f'visualization_step_{i}.html')
        
        # Generate and save screenshot
        hti.screenshot(
            url=html_path,
            save_as=f'visualization_step_{i}.png',
            size=(1200, 500) if type_txt == 'zhihu' else (1200, 800)
        )
