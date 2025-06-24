import sys


def error_message_detail(error: Exception) -> str:
    """
    Constructs a detailed error message including the file name, line number,
    and error description.

    Args:
        error (Exception): The exception that was raised.

    Returns:
        str: A formatted string containing the error details.
    """
    exc_type, exc_value, exc_tb = sys.exc_info()  # Extract the traceback information
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name = "<unknown>"
        line_no = 0
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_no}] error message [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """
    Custom exception class that provides detailed error messages.

    This class extends the base Exception class and uses the
    `error_message_detail` function to generate a comprehensive error message,
    including the file name and line number where the error occurred.

    Attributes:
        error_message (str): The formatted error message.
    """

    def __init__(self, error: Exception):
        """
        Initialize the CustomException instance.

        Args:
            error (Exception): The exception that was raised.
        """
        super().__init__(str(error))
        self.error_message = error_message_detail(error)

    def __str__(self) -> str:
        """
        Return the string representation of the error message.

        Returns:
            str: The detailed error message.
        """
        return self.error_message
