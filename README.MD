# edge-surf-pid

## Description

This project uses OpenCV and PyInput to create a bot that can play the Edge Surf game automatically, without the need for AI or machine/reinforcement learning models.

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

This will launch the bot and start playing the game automatically once the game at edge://surf is visible on the screen and started.

## How it Works

1. The bot uses simple color detection to detect the color of the obstacles in front of the surfer and then moves the surfer in the direction with lesser obstacles. 
2. The bot also uses a PID controller to control the extent of the surfer's movement.

By default, the bot will display a live feed of the game as it plays. You can disable this by setting the `SHOULD_SHOW_LIVE_FEED` variable to `False` in the `__main__.py` file.

## License

This project is licensed under the MIT License. See the [LICENSE](https://raw.githubusercontent.com/Dhi13man/edge-surf-pid/main/LICENSE) file for more information.
