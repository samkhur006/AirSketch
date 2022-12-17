import os
import pandas as pd
import math


class Rubine_feature_extractor(object):
    def __init__(self, df):
        self.df = df

    def f01(self, df):
        x0, x2 = df.loc[0, 'x'], df.loc[2, "x"]
        y0, y2 = df.loc[0, 'y'], df.loc[2, "y"]

        cos_a = (x2 - x0) / math.sqrt(pow(y2 - y0, 2) + pow(x2 - x0, 2))
        return cos_a

    def f02(self, df):
        x0, x2 = df.loc[0, 'x'], df.loc[2, "x"]
        y0, y2 = df.loc[0, 'y'], df.loc[2, "y"]

        sin_a = (y2 - y0) / math.sqrt(pow(y2 - y0, 2) + pow(x2 - x0, 2))
        return sin_a

    def f03(self, df):
        xmin, xmax = df["x"].min(), df["x"].max()
        ymin, ymax = df["y"].min(), df["y"].max()
        bounding_box_diagonal_dist = math.sqrt(pow(xmin - xmax, 2) + pow(ymin - ymax, 2))
        return bounding_box_diagonal_dist

    def f04(self, df):
        xmin, xmax = df["x"].min(), df["x"].max()
        ymin, ymax = df["y"].min(), df["y"].max()
        bounding_box_angle = math.atan2(ymax - ymin, xmax - xmin)
        return bounding_box_angle

    def f05(self, df):
        n = df.shape[0]
        x0, xn_1 = df.loc[0, "x"], df.loc[n - 1, "x"]
        y0, yn_1 = df.loc[0, "y"], df.loc[n - 1, "y"]

        endpoint_distance = math.sqrt(pow(xn_1 - x0, 2) + pow(yn_1 - y0, 2))
        return endpoint_distance

    def f06(self, df):
        n = df.shape[0]
        x0, xn_1 = df.loc[0, "x"], df.loc[n - 1, "x"]

        cos_angle_bw_endpoints = (xn_1 - x0) / (self.f05(df))
        return cos_angle_bw_endpoints

    def f07(self, df):
        n = df.shape[0]
        y0, yn_1 = df.loc[0, "y"], df.loc[n - 1, "y"]

        cos_angle_bw_endpoints = (yn_1 - y0) / (self.f05(df))
        return cos_angle_bw_endpoints

    def f08(self, df):
        n = df.shape[0]
        stroke_length = 0
        for i in range(0, n - 1):
            x0, x1 = df.loc[i, "x"], df.loc[i + 1, "x"]
            y0, y1 = df.loc[i, "y"], df.loc[i + 1, "y"]
            stroke_length += math.sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2))

        return stroke_length

    def f09(self, df):
        n = df.shape[0]
        total_angle = 0
        for i in range(0, n - 2):
            x0, x1, x2 = df.loc[i, "x"], df.loc[i + 1, "x"], df.loc[i + 2, "x"]
            y0, y1, y2 = df.loc[i, "y"], df.loc[i + 1, "y"], df.loc[i + 2, "y"]

            dx0, dx1 = (x1 - x0), (x2 - x1)
            dy0, dy1 = (y1 - y0), (y2 - y1)

            d_theta = math.atan2((dx1 * dy0 - dx0 * dy1), (dx1 * dx0 + dy1 * dy0))
            total_angle += d_theta

        return total_angle

    def f10(self, df):
        n = df.shape[0]
        total_absolute_angle = 0
        for i in range(0, n - 2):
            x0, x1, x2 = df.loc[i, "x"], df.loc[i + 1, "x"], df.loc[i + 2, "x"]
            y0, y1, y2 = df.loc[i, "y"], df.loc[i + 1, "y"], df.loc[i + 2, "y"]

            dx0, dx1 = (x1 - x0), (x2 - x1)
            dy0, dy1 = (y1 - y0), (y2 - y1)

            d_theta = math.atan2((dx1 * dy0 - dx0 * dy1), (dx1 * dx0 + dy1 * dy0))
            total_absolute_angle += abs(d_theta)

        return total_absolute_angle

    def f11(self, df):
        n = df.shape[0]
        total_squared_angle = 0
        for i in range(0, n - 2):
            x0, x1, x2 = df.loc[i, "x"], df.loc[i + 1, "x"], df.loc[i + 2, "x"]
            y0, y1, y2 = df.loc[i, "y"], df.loc[i + 1, "y"], df.loc[i + 2, "y"]

            dx0, dx1 = (x1 - x0), (x2 - x1)
            dy0, dy1 = (y1 - y0), (y2 - y1)

            d_theta = math.atan2((dx1 * dy0 - dx0 * dy1), (dx1 * dx0 + dy1 * dy0))
            total_squared_angle += pow(d_theta, 2)

        return total_squared_angle


    def all_features(self, df):
        df = (df-df.mean())/df.std()
        df_out = {"f01": self.f01(df), "f02": self.f02(df), "f03": self.f03(df), "f04": self.f04(df), "f05": self.f05(df), "f06": self.f06(df),
                  "f07": self.f07(df), "f08": self.f08(df), "f09": self.f09(df), "f10": self.f10(df), "f11": self.f11(df)}

        return df_out
