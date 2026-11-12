# AFM-RL Demo Input Folder

Drop any two images here that you want to match:
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.ppm`, `.bmp`
- Example names: `image1.jpg` and `image2.jpg` (or any names, sorted alphabetically)

Then simply run:
```bash
python scripts/demo_pair.py
```
Or use the GUI file selector:
```bash
python scripts/demo_pair.py --gui
```
Or pass specific paths:
```bash
python scripts/demo_pair.py --img1 "path/to/img1.jpg" --img2 "path/to/img2.jpg"
```
