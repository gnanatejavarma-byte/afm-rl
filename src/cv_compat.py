import cv2

def _search_spaces():
    spaces = [cv2]
    xf = getattr(cv2, "xfeatures2d", None)
    if xf is not None:
        spaces.append(xf)
    return spaces

def make(name, *args, **kwargs):
    """Create an OpenCV algorithm by name.
    Tries, in order, for the main cv2 namespace and then cv2.xfeatures2d:
      1) NAME_create(...)   (OpenCV 4.x style, also used by 5.0 in xfeatures2d)
      2) NAME.create(...)   (class-style factory)
    """
    for space in _search_spaces():
        legacy = getattr(space, f"{name}_create", None)
        if legacy is not None:
            return legacy(*args, **kwargs)
        cls = getattr(space, name, None)
        if cls is not None and hasattr(cls, "create"):
            return cls.create(*args, **kwargs)
    raise AttributeError(f"OpenCV has no factory for {name}")