# OpenClaw Scripts

Organized collection of photography workflow and automation scripts for Ronnie (Ron P. Wilder).

Tool map and status (active / paused / dropped / ideas): [`tools_tree.md`](tools_tree.md).
Tool reference docs: `CLAUDE.md`.

## Structure

*   **`workflows/`**: Stable and production-ready photography processing pipelines (e.g., Tensor Art stylization).
*   **`identity/`**: Scripts focusing on Face Swapping, InstantID, and identity preservation.
*   **`google_tools/`**: Automation for Google Calendar, GDrive, and Gmail.
*   **`utils/`**: General image processing utilities (EXIF, resizing, padding).
*   **`experiments/`**: Historical tests, one-off experiments, and older versions of scripts.

## Setup

These scripts depend on environment variables for authentication:
*   `TENSOR_API_KEY`
*   `FAL_API_KEY`
*   `GOOGLE_API_KEY`
*   `GITHUB_API_KEY`
