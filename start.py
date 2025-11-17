#!/usr/bin/env python3

import os
import sys
import subprocess
import re
import platform
import socket
import time

def check_os():
    platform_os = 'not_supported'
    if sys.platform.startswith('linux'):
        platform_os = 'linux'
    elif sys.platform == 'darwin':
        platform_os = 'macos'
    elif sys.platform == 'win32' or sys.platform == 'cygwin':
        platform_os = 'windows'

    return platform_os

def run_command(command):
    try:
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        if e.stderr:
            error_message = e.stderr.strip()
        else:
            error_message = str(e)
        print("Error:", error_message)
        sys.exit(1)

    return result

# Set no_proxy
if platform.system() == "windows":
    os.system('set no_proxy=localhost,127.0.0.0')
else:
    os.environ["no_proxy"] = "localhost,127.0.0.0"
print("Set no_proxy to localhost,127.0.0.0")

### Dynamic port allocation
def is_port_in_use(port):
    platform_os = check_os()
    if platform_os in ['linux', 'macos']:
        # Using lsof to check if the port is being used
        result = subprocess.run(["lsof", "-i", f":{port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stdout:
            print(f"Port {port} is in use. Details: {result.stdout.decode('utf-8')}")
        return result.returncode == 0  # If return code is 0, the port is in use
    elif platform_os == 'windows':
        # Using netstat to check if the port is being used
        result = subprocess.run(['netstat', '-ano', '-p', 'TCP'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stdout:
            lines = result.stdout.decode('utf-8').splitlines()
            for line in lines:
                if str(port) in line:
                    print(f"Port {port} is in use. Line: {line}")
                    return True
        return False
    else:
        print(f"Unsupported OS: {platform_os}")
        return False

def find_available_port(start_port=49152, end_port=65535):
    for port in range(start_port, end_port):
        if not is_port_in_use(port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
    print("No available ports found in the range 49152-65535.")
    sys.exit(1)

def main():
    try:
        # Parameters
        VERSION="1.7.0_Palette_SDK_master_B219"
        IMAGE = "palettesdk"
        CONTAINER_NAME = f"{IMAGE}_{VERSION.replace('.', '_')}"
        PORT = find_available_port()
        print(f"Using port {PORT} for the installation.")
        platform_os = check_os()
        # Checking if docker daemon is running
        if not (
            subprocess.run(
                ["docker", "ps"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).returncode
            == 0
        ):
            user_response = input(
                "Docker daemon is not running. Do you want to start it? [y/n]: "
            )

            if user_response.lower() == "y":
                print("Attempting to start docker service...")

                if platform_os == "linux":
                    docker_start_cmd = [
                        "sudo",
                        "systemctl",
                        "restart",
                        "docker.service",
                    ]
                elif platform_os == "windows":
                    print(
                        f"Starting Docker desktop on Windows requires user permission, please select \"start\" button on the Dialog box if prompted..."
                    )
                    docker_start_cmd = 'cmd.exe /c "start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe""'
                retry_count = 0
                start_flag = False

                if platform_os == "linux":
                    while retry_count < 3 and not start_flag:
                        start_flag = (
                            subprocess.run(
                                docker_start_cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            ).returncode
                            == 0
                        )
                        retry_count += 1
                        time.sleep(5)

                    if retry_count >= 3:
                        print("Failed to start docker service after 3 tries.")
                        print(
                            "If you have Docker installed, make sure it is running and try again."
                        )
                        sys.exit(1)
                elif platform_os == "windows":
                    start_flag = (
                        subprocess.run(
                            docker_start_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        ).returncode
                        == 0
                    )
                    time.sleep(5)
                    user_response = input(
                        "Attempted to start Docker Desktop, press RETURN[Enter] after it starts to proceed. Or Enter [q] to quit."
                    )
                    if user_response == "":
                        print(
                            "Received user confirmation to proceed, checking if docker daemon is running..."
                        )
                        if not (
                            subprocess.run(
                                ["docker", "ps"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            ).returncode
                            == 0
                        ):
                            print(
                                f"Docker daemon has not started, please contact your admin for assistance."
                            )
                            exit(1)
                    elif user_response.lower() == "q":
                        print(f"User has selected to quit installation.")
                        exit(1)
                    else:
                        print("Invalid response, exiting.")
                        exit(1)

            else:
                print(f"Please restart docker daemon and try again.")
                exit(1)

        # Checking if required SDK image is available
        containers_image = subprocess.check_output(["docker", "images", f"{IMAGE}:{VERSION}", "--format", "{{.Tag}}"], text=True).strip()
        if VERSION not in containers_image:
            print(f"SDK version {VERSION} not found. Please make sure to install it before proceeding to start the SDK.")
            print("You can install the SDK by running ./install.py or python3 install.py. Once installed, you can start the SDK using this script.")
            sys.exit(1)

        print("Checking if the container is already running...")
        existing_containers = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).split('\n')

        if CONTAINER_NAME in existing_containers:
            print(" ==> Container is already running. Proceeding to start an interactive shell.")
        else:
            stopped_containers = subprocess.check_output(["docker", "ps", "-a", "--format", "{{.Names}}"], text=True).split('\n')

            if CONTAINER_NAME in stopped_containers:
                print(f"  ==> Starting the stopped container: {CONTAINER_NAME}")
                subprocess.run(["docker", "start", CONTAINER_NAME], check=True)
            else:
                # Check if the port is in use
                netstat_result = subprocess.run(["netstat", "-tuln"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                if netstat_result.stdout is not None:
                    # Check for an exact match using regular expression
                    pattern = r"\b" + str(PORT) + r"\b"
                    if re.search(pattern, netstat_result.stdout.decode("utf-8")):
                        print(f"Required port {PORT} is currently in use.")
                        print("Please ensure that you stop any processes utilizing this port before proceeding with the SDK installation.")
                        sys.exit(1)

                home = os.path.expanduser("~")
                file_path = os.path.join(home, ".simaai", ".mount")

                if os.path.isfile(file_path):
                    with open(file_path, 'r') as file:
                        mount_path = file.read().strip()
                        if not mount_path:
                            mount_path = os.path.join(home, "workspace")
                else:
                    mount_path = os.path.join(home, "workspace")

                while True:
                    workspace = (
                        input(f"Enter work directory [{mount_path}]: ").strip() or mount_path
                    )
                    workspace = os.path.expanduser(workspace)
                    workspace = os.path.realpath(workspace)

                    if os.path.isdir(workspace):
                        if not os.path.exists(f"{home}/.simaai/"):
                            os.makedirs(f"{home}/.simaai/")

                        with open(file_path, "w") as f:
                            f.write(workspace)
                        break
                    else:
                        print(
                            "The specified work directory does not exist or is not a valid directory."
                        )
                        user_choice = (
                            input(
                                "Would you like to create the directory? (y/n) or press 'r' to retry with another directory: "
                            )
                            .strip()
                            .lower()
                        )

                        if user_choice == "y":
                            try:
                                os.makedirs(workspace)
                                print(f"Directory '{workspace}' has been created.")
                                if not os.path.exists(f"{home}/.simaai/"):
                                    os.makedirs(f"{home}/.simaai/")

                                with open(file_path, "w") as f:
                                    f.write(workspace)
                                break
                            except Exception as e:
                                print(f"Failed to create the directory. Error: {e}")
                        elif user_choice == "r":
                            continue
                        else:
                            print("Exiting the setup.")
                            sys.exit(0)

                print(f"Starting the container: {CONTAINER_NAME}")
                # Check SiMa SDK Bridge Network
                print("Checking SiMa SDK Bridge Network...")
                docker_networks_result = subprocess.run(["docker", "network", "ls", "--format", "{{.Name}}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                if "simasdkbridge" in docker_networks_result.stdout.decode("utf-8"):
                    print("SiMa SDK Bridge Network found.")
                else:
                    subprocess.run(["docker", "network", "create", "simasdkbridge"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if platform_os == 'linux' or platform_os == 'macos':
                    # Finding current user id & name on host and adding to Docker container
                    uid = os.getuid()
                    gid = os.getgid()
                    login_name = os.getlogin().replace(" ", "")
                else:
                    uid = 900
                    gid = 900
                    login_name = 'docker'

                # Create and start the Docker container
                print("Creating and starting the Docker container...")
                # Run the Docker container
                run_command([
                    "docker", "run", "-t", "-d", f"--user={uid}:{gid}", "--privileged",
                    "--cap-add=NET_RAW", "--cap-add=NET_ADMIN", "--name", CONTAINER_NAME, "-p", f"{PORT}:8084",
                    "--network", "simasdkbridge", "-v", f"{workspace}:/home/docker/sima-cli/", f"{IMAGE}:{VERSION}"
                ])

                if platform_os == 'linux' or platform_os == 'macos':
                    # Copy and update /etc/passwd
                    run_command(["docker", "cp", f"{CONTAINER_NAME}:/etc/passwd", "./passwd.txt"])
                    with open("./passwd.txt", "a") as f:
                        f.write(f"{login_name}:x:{uid}:{gid}::/home/{login_name}:/bin/sh\n")
                    run_command(["docker", "cp", "./passwd.txt", f"{CONTAINER_NAME}:/etc/passwd"])
                    os.remove("./passwd.txt")

                    # Copy and update /etc/shadow
                    run_command(["docker", "cp", f"{CONTAINER_NAME}:/etc/shadow", "./shadow.txt"])
                    with open("./shadow.txt", "a") as f:
                        f.write(f"{login_name}:$6$ZlXYuKGiYsynU/ff$Irnq0VW7sMSl1VKCa/hdJM/Oq/3wqHm8YOcfRrnH1tb2w.r/1P4Z4NrHiBhIStdnF4A/tmhGY0Xh.c/cr1hmE0:19489:0:99999:7:::\n")
                    run_command(["docker", "cp", "./shadow.txt", f"{CONTAINER_NAME}:/etc/shadow"])
                    os.remove("./shadow.txt")

                    # Copy and update /etc/group
                    run_command(["docker", "cp", f"{CONTAINER_NAME}:/etc/group", "./group.txt"])
                    with open("./group.txt", "a") as f:
                        f.write(f"{login_name}:x:{gid}:\n")
                    run_command(["docker", "cp", "./group.txt", f"{CONTAINER_NAME}:/etc/group"])
                    os.remove("./group.txt")

                    # Enabling current user as passwordless sudo
                    run_command(["docker", "cp", f"{CONTAINER_NAME}:/etc/sudoers", "./sudoers.txt"])
                    os.chmod("./sudoers.txt", 0o755)
                    with open("./sudoers.txt", "a") as f:
                        f.write(f"{login_name} ALL=(ALL:ALL) NOPASSWD:ALL\n")
                    run_command(["docker", "cp", "./sudoers.txt", f"{CONTAINER_NAME}:/etc/sudoers"])
                    run_command(["docker", "exec", "-u", "root", f"{CONTAINER_NAME}", "chmod", "440", "/etc/sudoers"])
                    run_command(["docker", "exec", "-u", "root", f"{CONTAINER_NAME}", "chown", "root:root", "/etc/sudoers"])
                    os.remove("./sudoers.txt")

                    # Adding the current user to the Docker group
                    run_command(["docker", "exec", "-u", "root", CONTAINER_NAME, "usermod", "-a", "-G", "docker", login_name])

                    # Creating a home directory for the local user inside the Docker container
                    home_directory = f"/home/{login_name}"
                    run_command(["docker", "exec", "-u", "root", CONTAINER_NAME, "mkdir", "-p", home_directory])
                    run_command(["docker", "exec", "-u", "root", CONTAINER_NAME, "chown", f"{uid}:{gid}", home_directory])
                else:
                    # Define the command to execute inside the container
                    command = f"echo '{login_name} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers"
                    run_command(["docker", "exec", "-u", "root", f"{CONTAINER_NAME}", "sh", "-c", f"{command}"])

                # Save custom port information inside the container
                with open("./.port", 'w') as file:
                    file.write(str(PORT))
                run_command(["docker", "cp", "./.port", f"{CONTAINER_NAME}:/home/docker/.simaai/.port"])
                os.remove("./.port")

                run_command(["docker", "exec", "-u", "root", CONTAINER_NAME, "chown", "-R", f"{uid}:{gid}", "/usr/local/simaai/"])

                # Add the current user to rsyslog config
                run_command(["docker", "exec", "-u", "root", CONTAINER_NAME, "sed", "-i.bk", f"s@docker@{login_name}@g", "/etc/rsyslog.conf"])

                run_command(["docker", "exec", "-u", "root", CONTAINER_NAME, "chown", "-R", f"{uid}:{gid}", "/usr/local/simaai/"])

        # Start docker container
        run_command(["docker", "exec", "-it", CONTAINER_NAME, "bash"])
        
    except Exception as e:
        if e.stderr:
            error_message = e.stderr.strip()
        else:
            error_message = str(e)
        print("Error:", error_message)

if __name__ == "__main__":
    main()
