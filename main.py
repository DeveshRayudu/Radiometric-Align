"""
main

Top-level launcher for the Raster Align desktop app.

Starts the PyQt5 GUI (gui.gui_main.main), which implements the flow:
Select Source Image -> Select Target Image -> Click Start -> Validate
Inputs -> Create outputs/ -> Create Timestamp Folder -> Start Background
Worker -> Invoke Existing Pipeline (raster_align.run_pipeline) ->
Processing Modules -> Save Outputs -> Write Logs -> Notify GUI ->
Pipeline Completed.

The command-line interface is still available separately via
`python -m raster_align.cli` for scripted/headless runs; it is not
invoked from here.
"""

from gui.gui_main import main

if __name__ == "__main__":
    main()