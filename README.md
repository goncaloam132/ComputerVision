# Computer Vision - Traffic and Object Detection with YOLOv8

A Python-based computer vision project that detects, counts, and annotates vehicles and other objects in images using the YOLOv8 model. This project provides a complete pipeline for object detection, including inference, bounding box extraction, class counting, and annotated image generation.

## Features

- 🚗 **Vehicle Detection**: Detects cars, trucks, motorcycles, and buses
- 📊 **Object Counting**: Counts instances by class with detailed statistics
- 🖼️ **Image Annotation**: Draws bounding boxes and labels on detected objects
- 📁 **Batch Processing**: Process single images or entire directories
- 💾 **Multiple Export Formats**: Save results as CSV, JSON, or annotated images
- 🎯 **Filtering Options**: Focus on specific object classes (e.g., only vehicles)

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/goncaloam132/ComputerVision.git
cd ComputerVision
pip install -r requirements.txt
```

**Requirements:**  
- Python 3.8+
- [ultralytics](https://pypi.org/project/ultralytics/) (YOLOv8)
- opencv-python
- numpy
- tqdm

## Quick Start

Process all images in the `imagens/` directory:

```bash
python detect_traffic.py -i imagens/
```

Process a single image with custom output:

```bash
python detect_traffic.py -i imagens/imagem0.jpg -o my_result.jpg --show
```

## Usage

### Command Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `-i`, `--input` | Input image(s) or directory path | ✅ | - |
| `-o`, `--output` | Output image path | ❌ | `resultados/output_<filename>` |
| `-m`, `--model` | YOLOv8 model path | ❌ | `yolov8n.pt` |
| `--csv` | CSV file path for counts | ❌ | - |
| `--json` | JSON file path for counts | ❌ | - |
| `--show` | Display annotated image | ❌ | False |
| `--only-vehicles` | Count only vehicles | ❌ | False |

### Examples

**Process all images in a directory:**
```bash
python detect_traffic.py -i imagens/ --csv results.csv --json results.json
```

**Detect only vehicles and show results:**
```bash
python detect_traffic.py -i imagens/ --only-vehicles --show
```

**Process single image with custom model:**
```bash
python detect_traffic.py -i test.jpg -m yolov8s.pt -o annotated.jpg
```

## Project Structure

```
ComputerVision/
├── detect_traffic.py        # Main detection and annotation script
├── imagens/                 # Test images for inference
├── resultados/              # Output annotated images
├── yolov8n.pt              # Pre-trained YOLOv8 model weights
├── requirements.txt         # Project dependencies
├── README.md               # This documentation
└── docs/
    └── approach.md         # Technical approach and challenges
```

## Output

The script generates:
- **Annotated Images**: Original images with bounding boxes and labels
- **Count Statistics**: Number of objects detected per class
- **Export Files**: Optional CSV/JSON files with detailed results

### Sample Output
```
Contagens para imagens/imagem0.jpg: {'car': 3, 'truck': 1, 'person': 2}
Resultado salvo em resultados/output_imagem0.jpg
```

## Technical Details

The project uses YOLOv8 (You Only Look Once version 8) for real-time object detection. The pipeline includes:

1. **Model Loading**: Loads pre-trained YOLOv8 weights
2. **Inference**: Processes images through the neural network
3. **Post-processing**: Extracts bounding boxes, confidence scores, and class IDs
4. **Visualization**: Draws annotations using OpenCV
5. **Export**: Saves results in multiple formats

## Approach & Challenges

Detailed technical information about the implementation, challenges faced, and solutions can be found in [docs/approach.md](docs/approach.md).

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## License

This project is open source and available under the [MIT License](LICENSE).

