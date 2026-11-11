Fixed
^^^^^

* Fixed RLinf checkpoint resolution so training could restore the model and
  sharded optimizer state from the runner checkpoint root.
* Updated embodied actor discovery and disabled Ray uv project uploads during evaluation.
* Fixed playback checkpoint forwarding to RLinf's rollout worker so evaluation
  loaded the selected trained policy instead of retaining the base model.
