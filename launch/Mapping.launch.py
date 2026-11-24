"""Launch the Airy LIO node; override config and use_sim_time for bag replay."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('ieskf_voxel_lio'), 'config', 'RobosenseAiry.yaml'
    ])
    default_rviz = PathJoinSubstitution([
        FindPackageShare('ieskf_voxel_lio'), 'rviz', 'IeskfLio.rviz'
    ])
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Start RViz with the LIO configuration'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        Node(
            package='ieskf_voxel_lio', executable='lio_node', output='screen',
            parameters=[LaunchConfiguration('config'),
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
        Node(
            package='rviz2', executable='rviz2', output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
