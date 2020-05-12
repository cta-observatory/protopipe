.. _install-developer:

Developers
==========

If you want to use *protopipe* and also contribute to its development, follow these steps:

  1. Fork the official `repository <https://github.com/cta-observatory/protopipe>`_ has explained `here <https://help.github.com/en/articles/fork-a-repo>`__ (follow all the instructions)
  2. now your local copy is linked to your remote repository (**origin**) and the official one (**upstream**)
  3. execute points 3 and 4 in the instructions for :ref:`install-basic`.
  4. install *protopipe* itself in developer mode with ``python setup.py develop``

In this way, When you change branch or modify the code, the scripts will always
refer to the that version of the code and you will not need to install it again.
