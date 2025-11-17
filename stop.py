#!/usr/bin/env python3

import subprocess
import sys

def main():
    try:
        # Check if Docker is available
        if not (subprocess.run(["docker", "ps"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0):
            print("Docker daemon is not running.")
            print("If you have Docker installed, make sure it is running and try again.")
            sys.exit(1)

        # Get list of running containers with name starting with "palettesdk"
        containers = subprocess.check_output(["docker", "ps", "-q", "--filter", "name=palettesdk*"], text=True).strip().split('\n')

        # Loop through list of running containers and prompt user to stop and remove each one
        for container_id in containers:
            if container_id:
                # Get container name from ID
                container_name = subprocess.check_output(["docker", "inspect", "--format", "{{.Name}}", container_id], text=True).strip()[1:]

                # Ask user if they want to stop and remove the container and its image
                stop_container = input(f"Do you want to stop '{container_name}' (y/n): ")

                if stop_container.lower() == 'y':
                    # Stop the container if the user chooses to stop it
                    subprocess.run(["docker", "stop", container_id], check=True)
                    print(f"Container {container_name} stopped.")
                    remove_container = input(f"Do you want to remove '{container_name}' (y/n): ")

                    if remove_container.lower() == 'y':
                        # Remove the container
                        subprocess.run(["docker", "rm", container_id], check=True)
                else:
                    # Skip the container if the user chooses not to stop it
                    print(f"Container {container_name} skipped.")
    except subprocess.CalledProcessError as e:
        if e.stderr:
            error_message = e.stderr.strip()
        else:
            error_message = str(e)
        print("Error:", error_message)

if __name__ == "__main__":
    main()
