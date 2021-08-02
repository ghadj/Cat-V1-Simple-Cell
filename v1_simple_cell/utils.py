#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np


def rotate_point(x, y, theta):
    """Rotate the given point (x, y) at an angle theta.

    Args:
        x (float): x coordinate.
        y (float): y coordinate.
        theta (float): orientation angle.

    Returns:
        (float, float): rotated x and y coordinates.
    """

    xx = x*np.cos(theta) - y*np.sin(theta)
    yy = x*np.sin(theta) + y*np.cos(theta)
    return xx, yy


def rotate_point_arr(xs, ys, theta):
    """Rotates multiple points at an angle theta.

    Args:
        xs (list of floats): x coordinates.
        ys (list of floats): y coordinates.
        theta (float): orientation angle.

    Returns:
        (lists of floats, lists of floats): two lists containing the 
            rotated x and y coordinates.
    """

    xs_rot = np.zeros(xs.size)
    ys_rot = np.zeros(ys.size)

    for i, (x, y) in enumerate(zip(xs, ys)):
        xs_rot[i], ys_rot[i] = rotate_point(x, y, theta)
    return xs_rot, ys_rot


def concat_arr(a, b, c, d):
    """Concatenates four lists into one.

    Args:
        a (list of float): a list to be concatenated.
        b (list of float): a list to be concatenated.
        c (list of float): a list to be concatenated.
        d (list of float): a list to be concatenated.

    Returns:
        list of float: the concatenation of the given lists.
    """

    e = np.concatenate((a, b), axis=0)
    f = np.concatenate((c, d), axis=0)
    return np.concatenate((e, f), axis=0)


def get_index_bounds(indexes):
    """Returns the bounds (first and last element) of consecutive numbers.

    Example:
    [1 , 3, 4, 5, 7, 8, 9, 10] -> [1, 3, 5, 7, 10]

    Args:
        indexes (list of int): integers in an ascending order.

    Returns:
        list of int: the bounds (first and last element) of consecutive numbers.
    """

    index_bounds = [indexes[0]]

    for i, v in enumerate(indexes):
        if i == len(indexes)-1:
            index_bounds.append(v)
            break

        if v+1 == indexes[i+1]:
            continue
        else:
            index_bounds.append(v)
            index_bounds.append(indexes[i+1])

    return index_bounds


def get_interval_indexes(interval, indexes):
    """Returns the given indexes, after removing <interval> elements from the
    beginning and ending of each sequence of consecutive numbers.

    Args:
        interval (int): number of elements to be removed from beginning and 
            ending of each sequence of consecutive numbers.
        indexes (list of int): integers in an ascending order.

    Returns:
        list of int: the given indexes, after removing <interval> elements from 
            the beginning and ending of each sequence of consecutive numbers.
    """

    if len(indexes) == 0:
        return 0

    index_bounds = get_index_bounds(indexes)

    interval_indexes = []

    for i in range(0, len(index_bounds)-1, 2):
        if index_bounds[i]+interval >= index_bounds[i+1]:
            continue

        interval_indexes.extend(range(index_bounds[i]+interval,
                                      index_bounds[i+1]-interval+1))

    return interval_indexes


if __name__ == "__main__":
    l = np.array([1, 3, 4, 5, 6, 8, 10, 11, 12])
    print(l)
    tmp = get_index_bounds(l)
    print(tmp)
    print(get_interval_indexes(2, l))
