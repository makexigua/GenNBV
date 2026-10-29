"""Compatibility entry for the earlier standalone LSD/DeepLSD tool."""
from chapter3.perception.legacy_lines import DeepLSDDetector, detect_lines, save_detection, main

if __name__ == "__main__":
    main()
