from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml
import os

def generate_launch_description():

    config_file_path = os.path.join(
        get_package_share_directory('imitation_nav'),
        'config',
        'params.yaml'
    )

    nav_node = Node(
        package = 'imitation_nav',
        executable = 'imitation_nav_node',
        parameters = [config_file_path],
        output = 'screen'
    )

    return LaunchDescription([nav_node])