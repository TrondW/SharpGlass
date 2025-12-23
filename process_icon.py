import sys
from PIL import Image, ImageDraw, ImageOps, ImageFilter

def create_macos_icon(input_path, output_path):
    try:
        # macOS Icon Spec
        # Canvas: 1024x1024
        # Icon Shape: ~824x824 (approx 80.5% of canvas)
        # Corner Radius: ~22% of the shape size
        
        CANVAS_SIZE = 1024
        SHAPE_SIZE = 824
        PADDING = (CANVAS_SIZE - SHAPE_SIZE) // 2
        
        # Load source
        img = Image.open(input_path).convert("RGBA")
        
        # 1. Create the Squircle Mask for the Content
        mask = Image.new('L', (SHAPE_SIZE, SHAPE_SIZE), 0)
        draw = ImageDraw.Draw(mask)
        
        # Squircle-ish radius
        radius = SHAPE_SIZE * 0.22 
        draw.rounded_rectangle([(0, 0), (SHAPE_SIZE, SHAPE_SIZE)], radius=radius, fill=255)
        
        # 2. Resize/Fit Source Image to Shape Size and Mask it
        content = ImageOps.fit(img, (SHAPE_SIZE, SHAPE_SIZE), centering=(0.5, 0.5))
        content.putalpha(mask)
        
        # 3. Create Drop Shadow
        # Shadow roughly same shape, blurred
        shadow_size = int(SHAPE_SIZE * 0.98) # Slightly smaller to hide edges
        shadow = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (0,0,0,0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # Shadow position: slightly lower
        shadow_offset_y = 10 
        shadow_rect = [
            (PADDING, PADDING + shadow_offset_y), 
            (PADDING + SHAPE_SIZE, PADDING + SHAPE_SIZE + shadow_offset_y)
        ]
        
        # Draw black rounded rect for shadow
        # Use a slightly softer radius for shadow
        shadow_draw.rounded_rectangle(shadow_rect, radius=radius, fill=(0,0,0,160)) # 60% opacity black
        
        # Blur the shadow
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))
        
        # 4. Composite
        final_canvas = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (0,0,0,0))
        final_canvas.alpha_composite(shadow)
        final_canvas.alpha_composite(content, (PADDING, PADDING))
        
        # Save high-res master
        final_canvas.save(output_path, "PNG")
        print(f"Created macOS-compliant icon at {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 process_icon.py <input> <output>")
        sys.exit(1)
    
    create_macos_icon(sys.argv[1], sys.argv[2])
