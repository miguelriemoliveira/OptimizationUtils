#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import pickle

def main():
    # -----------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------
    parser = argparse.ArgumentParser(description='PSR argparse example.')
    parser.add_argument('-o','--output', type=str, default='points.pkl', help='Filename to write.')
    args = vars(parser.parse_args())
    print(args)

    plt.plot(0,0)
    plt.grid()
    plt.axis([-5, 5, -5, 5])

    # -----------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------
    points = []
    while True:

        plt.title('There are ' + str(len(points)) + ' points')
        print("Click a point (press Esc to terminate)... ")
        clicked_point = plt.ginput(1)
        print(clicked_point)

        if not clicked_point:  # empty list, terminate
            print('Pressed Esc, terminating.')
            break
        else:  # add new point to list
            points.append({'x': clicked_point[0][0], 'y': clicked_point[0][1]})
        print(points)

        # list comprehension to get all x  and y coordinates in separate lists
        xs = [item['x'] for item in points]
        ys = [item['y'] for item in points]
        plt.plot(xs, ys, 'bo')
        plt.draw()

    # Save points to file
    with open(args['output'], 'wb') as file_handle:
        pickle.dump(points, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saved file to ' + str(args['output']))
    # with open('filename.pickle', 'rb') as file_handle:
    #     b = pickle.load(file_handle)

    # -----------------------------------------------------
    # TERMINATION
    # -----------------------------------------------------


if __name__ == '__main__':
    main()
