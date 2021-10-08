.. Substitutions
.. |cmd| unicode:: U+2318
.. |opt| unicode:: U+2325
.. |editor requirements| replace:: support for syntax-specific code coloring and syntax-specific formatting and there should be linting_ for Python and JSON built-in or available through add-on packages. Python code linting should include checking for compliance with `PEP 8`_ (using the `pycodestyle`_ package) and pyflakes_, at a minimum

.. _software-require:

Software Requirements
=====================

The latest version of the full FlexAssist code base can be `downloaded on GitHub`_. FlexAssist runs on Python and several of its supporting packages, as outlined below.

.. _downloaded on GitHub: https://github.com/jtlangevin/flex-bldgs/releases

**Prerequisites***

* Python 3
* Python packages: pymc3, theano, numpy, scipy, arviz, matplotlib

Instructions follow for installing Python and these supporting packages on Mac OS and Windows.

.. _qs-mac:

Mac OS installation
-------------------

0. (Optional) Install a package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Mac OS ships with Python already installed. Installing and using a package manager will make it easy to ensure that any additional installations of Python do not interfere with the version of Python included with the operating system. Homebrew_ is a popular package manager, but there are other options, such as MacPorts and Fink.

.. _Homebrew website:
.. _Homebrew: http://brew.sh

.. note::
   While this step is optional, subsequent instructions are written with the assumption that you have installed Homebrew as your package manager.

To install Homebrew, open Terminal (found in Applications/Utilities, or trigger Spotlight with |cmd|-space and type "Terminal"). Visit the `Homebrew website`_ and copy the installation command text on the page. Paste the text into the Terminal application window and press Return. If you encounter problems with the installation, return to the Homebrew website for help or search online for troubleshooting assistance.

If you are using a package manager other than Homebrew, follow the documentation for that package manager to install Python 3. If you have chosen to not install a package manager, you may use the `Python Software Foundation installer`_ for the latest version of Python 3.

.. _Python Software Foundation installer: https://www.python.org/downloads/

1. Install Python 3
~~~~~~~~~~~~~~~~~~~

In a Terminal window, at the command prompt (a line terminated with a $ character and a flashing cursor), type::

   brew install python3

2. Install required Python packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once Python 3 is fully installed, pip3 is the tool you will use to install add-ons specific to Python 3. Six packages, pymc3, theano, numpy, scipy, arviz, and matplotlib, are required for Scout. To install them, at the command prompt in Terminal, type::

   pip3 install pymc3 theano numpy scipy arviz matplotlib

If you'd like to confirm that the packages were installed successfully, you can start Python from the command prompt in Terminal by typing::

   python3

and import the packages (within the Python interactive shell, indicated by the ``>>>`` prompt). :: 

   import theano, numpy, scipy, arviz, matplotlib

If no error or warning messages appear, then the installation was successful and you can exit Python by typing ``quit()``.

.. _qs-windows:

Windows installation
--------------------

1. Install Python 3
~~~~~~~~~~~~~~~~~~~

.. tip::
   If you have 64-bit Windows installed on your computer, downloading and installing the 64-bit version of Python is recommended. 

Download the executable installer for Windows available on the Python Software Foundation `downloads page`_. Run the installer and follow the on-screen prompts as you would with any other software installer. Be sure that the option in the installer "Add Python 3.x to PATH," where x denotes the current version of Python 3, is checked.

.. _downloads page: https://www.python.org/downloads/


2. Install required Python packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once Python 3 installation is complete, six Python packages need to be installed: pymc3, theano, numpy, scipy, arviz, and matplotlib. pip is the tool you will use to install add-ons specific to Python 3. Begin by `opening a command prompt`_ window. At the prompt (a line of text with a file path terminated by a greater than symbol, such as ``C:\>``), type::

   py -3 -m pip install pymc3 theano numpy scipy arviz matplotlib

.. _Open a command prompt:
.. _opening a command prompt: http://www.digitalcitizen.life/7-ways-launch-command-prompt-windows-7-windows-8

If you would like to confirm that the packages were installed successfully, you can open an interactive session of Python in a command prompt window by typing::

   py -3

and then importing the packages (within the Python interactive session, indicated by a ``>>>`` prompt)::

   import pymc3, theano, numpy, scipy, arviz, matplotlib

If no error or warning messages appear, the packages were installed successfully. Exit the interactive session of Python by typing::

   quit()
