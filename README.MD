# edge-surf-pid

## Description

This is a Python project that uses OpenCV and PyInput to create a bot that can play the Edge Surf game automatically, without the need for AI or machine learning algorithms.

## Requirements

To run this project, you will need to have the following dependencies installed:

- Python 3.6 or higher
- OpenCV
- MSS (Multi-Screen Shot)
- NumPy
- Simple-PID
- Pynput

You can install these dependencies using pip by running the following command:

```sh
pip install -r requirements.txt
```

## Usage

To use the Edge Surf bot, simply add the central surfer character's position in `__main__.py` and then run the `__main__.py` file using Python:

```sh
python __main__.py
```

This will launch the bot and start playing the game automatically. The bot uses simple color detection to detect the color of the next platform and then moves the surfer in the corresponding direction. The bot also uses a PID controller to control the extent of the surfer's movement.

By default, the bot will display a live feed of the game as it plays. You can disable this by setting the `SHOULD_SHOW_LIVE_FEED` variable to `False` in the `__main__.py` file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
