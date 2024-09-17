import numpy as np
import cv2

COLORS = {
    "BLACK": (0, 0, 0),
    "WHITE": (220, 220, 220),
    "BG_GREEN": (45, 90, 20),
    "ROBOT_BLACK": (25, 25, 25),
    "ORANGE": (2, 106, 253),
    "BLUE": (255, 64, 0),
    "YELLOW": (94, 218, 250),
    "GREEN": (20, 220, 57),
    "RED": (0, 21, 151),
    "PURPLE": (153, 51, 102),
    "PINK": (220, 0, 220),
}

class Robot:
    def __init__(self,
                 x_mm: float = 0,
                 y_mm: float = 0,
                 theta_rad: float = 0,
                 diameter_mm: float = 180):
        self.x = x_mm
        self.y = y_mm
        self.theta = theta_rad
        self.radius = diameter_mm/2

class LineSegment2D:
    def __init__(self,
                 x1: float,
                 y1: float,
                 x2: float,
                 y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    @property
    def coef(self):
        return np.arctan2(self.y2-self.y1, self.x2-self.x1)
    
    def mul(self, value):
        self.x1 *= value
        self.y1 *= value
        self.x2 *= value
        self.y2 *= value

class Goal:
    def __init__(self,
                 goal_depth = 180,
                 goal_width = 1000):
        self.UpperPostBar = LineSegment2D(x1=3900, y1=goal_width/2, x2=3900+goal_depth, y2=goal_width/2)
        self.LowerPostBar = LineSegment2D(x1=3900, y1=-goal_width/2, x2=3900+goal_depth, y2=-goal_width/2)
        self.BackBar = LineSegment2D(x1=3900+goal_depth, y1=-goal_width/2, x2=3900+goal_depth, y2=goal_width/2)

class FieldMarkings:
    def __init__(self):
        self.TopTouchLine = LineSegment2D(x1=0, y1=2700, x2=3900, y2=2700)
        self.BottomTouchLine = LineSegment2D(x1=0, y1=-2700, x2=3900, y2=-2700)
        self.RightGoalLine = LineSegment2D(x1=3900, y1=2700, x2=3900, y2=-2700)
        self.HalfwayLine = LineSegment2D(x1=0, y1=-2700, x2=0, y2=2700)
        self.CenterLine = LineSegment2D(x1=0, y1=0, x2=3900, y2=0)
        self.RightPenaltyStretch = LineSegment2D(x1=3000, y1=-900, x2=3000, y2=900)
        self.RightFieldRightPenaltyStretch = LineSegment2D(x1=3900, y1=-900, x2=3000, y2=-900)
        self.RightFieldLeftPenaltyStretch = LineSegment2D(x1=3900, y1=900, x2=3000, y2=900)

class FieldBoundaries:
    def __init__(self):
        self.TopBoundaryLine = LineSegment2D(x1=-300, y1=3000, x2=4200, y2=3000)
        self.BottomBoundaryLine = LineSegment2D(x1=-300, y1=-3000, x2=4200, y2=-3000)
        self.RightBoundaryLine = LineSegment2D(x1=4200, y1=3000, x2=4200, y2=-3000)
        self.LeftBoundaryLine = LineSegment2D(x1=-300, y1=3000, x2=-300, y2=-3000)

    def get_field_dimensions(self):
        x_min = min(self.TopBoundaryLine.x1, self.TopBoundaryLine.x2,
                    self.BottomBoundaryLine.x1, self.BottomBoundaryLine.x2,
                    self.RightBoundaryLine.x1, self.RightBoundaryLine.x2,
                    self.LeftBoundaryLine.x1, self.TopBoundaryLine.x2)
        x_max = max(self.TopBoundaryLine.x1, self.TopBoundaryLine.x2,
                    self.BottomBoundaryLine.x1, self.BottomBoundaryLine.x2,
                    self.RightBoundaryLine.x1, self.RightBoundaryLine.x2,
                    self.LeftBoundaryLine.x1, self.TopBoundaryLine.x2)

        y_min = min(self.TopBoundaryLine.y1, self.TopBoundaryLine.y2,
                    self.BottomBoundaryLine.y1, self.BottomBoundaryLine.y2,
                    self.RightBoundaryLine.y1, self.RightBoundaryLine.y2,
                    self.LeftBoundaryLine.y1, self.TopBoundaryLine.y2)
        y_max = max(self.TopBoundaryLine.y1, self.TopBoundaryLine.y2,
                    self.BottomBoundaryLine.y1, self.BottomBoundaryLine.y2,
                    self.RightBoundaryLine.y1, self.RightBoundaryLine.y2,
                    self.LeftBoundaryLine.y1, self.TopBoundaryLine.y2)
        
        length = x_max - x_min
        width  = y_max - y_min

        return length, width

class Render:
    def __init__(self,
                 img_height,
                 field_length_mm,
                 field_width_mm,
                 boundary_width_mm,
                 field_origin_mm):
        
        self.make_background(img_height, field_length_mm, field_width_mm)
        self.img = self.background.copy()

        self.scale = self.img.shape[1]/field_length_mm
        self.boundary_width = self.scale*boundary_width_mm
        self.center_offset = self.scale*np.array(field_origin_mm)
        self.markings = FieldMarkings()
        self.goal = Goal()

        self.draw_goal()
        self.draw_field_markings()

    def reset(self):
        self.img = self.background.copy()
        self.draw_goal()
        self.draw_field_markings()

    def make_background(self, img_height, field_length, field_width):
        field_proportion = field_length/field_width
        img_width = int(field_proportion*img_height)
        self.background = np.full((img_height, img_width, 3), COLORS['BG_GREEN'], dtype=np.uint8)

    def convert_field_to_image_coordinates(self, x_mm, y_mm):
        u, v = (self.scale*np.array([x_mm, -y_mm]) + self.center_offset).astype(int)
        return u, v
    
    def field2image(self, x_mm, y_mm):
        return self.convert_field_to_image_coordinates(x_mm, y_mm)

    def draw_line_segment(self, color = COLORS['WHITE'], thickness = 2, segment: LineSegment2D = None):
        u1, v1 = self.field2image(segment.x1, segment.y1)
        u2, v2 = self.field2image(segment.x2, segment.y2)
        cv2.line(self.img, (u1, v1), (u2, v2), color, thickness)
    
    def draw_field_markings(self):
        self.draw_line_segment(segment=self.markings.BottomTouchLine)
        self.draw_line_segment(segment=self.markings.TopTouchLine)
        self.draw_line_segment(segment=self.markings.CenterLine)
        self.draw_line_segment(segment=self.markings.HalfwayLine)
        self.draw_line_segment(segment=self.markings.RightGoalLine)
        self.draw_line_segment(segment=self.markings.RightFieldRightPenaltyStretch)
        self.draw_line_segment(segment=self.markings.RightFieldLeftPenaltyStretch)
        self.draw_line_segment(segment=self.markings.RightPenaltyStretch)
    
    def draw_goal(self):
        self.draw_line_segment(segment=self.goal.UpperPostBar)
        self.draw_line_segment(segment=self.goal.LowerPostBar)
        self.draw_line_segment(segment=self.goal.BackBar)
    
    def draw_robot(self, color = COLORS['BLUE'], thickness: int = 2, robot: Robot = None):
        radius = int(self.scale*robot.radius)
        u, v = self.field2image(robot.x, robot.y)
        u_r, v_r = int(u + radius*np.cos(-robot.theta)), int(v + radius*np.sin(-robot.theta))
        cv2.circle(self.img, (u, v), radius, color, -1)
        cv2.line(self.img, (u, v), (u_r, v_r), COLORS["BLACK"], thickness)
    
    def imshow(self, title='render'):
        cv2.imshow(title, self.img)

if __name__ == "__main__":
    img_height = 900
    boundary_width = 300
    field_length, field_width = FieldBoundaries().get_field_dimensions()

    render = Render(img_height, field_length, field_width, boundary_width, (boundary_width, field_width/2))

    render.draw_robot(robot=Robot(3000, 1000, np.rad2deg(45)))

    render.imshow()

    cv2.waitKey(0)

    cv2.destroyAllWindows()
