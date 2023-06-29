# Toolbox for PN Reduced Kinetic Models

## Description
TODO

## Installation and Running

To use this project on your host system, follow these steps:

1. Ensure that the following dependencies are installed on your system:

    | Name         | Version | Description                                                                                                                   |
    | ------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
    | `Python`     | >=3.7   | Required for running and calling the main code. Make sure to have `pip` and `venv` installed as well.                         |

    **Note**: If you're working in a cluster environment, load the appropriate environment modules to make the required packages available.
 
2. Clone the repository

    ```bash
    git clone https://github.com/p-gerhard/pntools.git
    ``` 
      
3. Create a virtual environment for the project:

    ```bash
    python3 -m venv venv-pntools
      ```

4. Activate the virtual environment:

    ```bash
    source venv-pntools/bin/activate
    ```
    
5. Install the package

    ```bash
    pip install -r requirement.txt
    ``` 
    
6. Run the code

    ```bash
    python3 pntools.py
    ``` 
