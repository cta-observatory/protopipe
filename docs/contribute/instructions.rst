.. _instructions:

Instructions
============

These are some guidelines on how to contribute to *protopipe* through its
repository in GitHub.
This of course makes sense only for the development branch, aka the *master*
branch.

This is usually done is 4 steps:

1. you start using protopipe
2. you find that either there is problem or *protopipe*
   is missing a feature that important for your research
3. you open an issue
4. you open a pull-request

Open an issue
-------------

Technically you could open directly a pull-request, but is preferable to open an
issue first, in order to warn others and trigger a possible discussion.
This will be useful also to identify more precisely what needs to be done.

If you are not able to do it, the administrators of the repository should **label
your issue** depending on its nature.
The labels most used in protopipe are quite self-explanatory:

- bug
- fix
- wrong behaviour
- enhancement
- documentation
- dependency update
- summary

If you find that these are limited, you can propose new ones.

Prepare your pull-request
-------------------------

This section assumes that you went through the installation for developers.

When you want to fix a bug or develop something new:

  1. update your **local** *master* branch with `git pull upstream master`
  2. create and move to a new **local** branch from your **local** *master* with
     `git checkout -b your_branch`
  3. develop inside it
  4. push it to *origin*, thereby creating a copy of your branch also there
  5. continue to develop and push until you feel ready
  6. start a *pull request* using the web interface from *origin/your_branch*
     to *upstream/master*

    1. wait for an outcome
    2. if necessary, you can update or fix things in your branch because now
       everything is traced
       (**local/your_branch** --> **origin/your_branch** --> **pull request**)

.. Note::

  If your developments take a relatively long time, consider to update periodically your **local** *master* branch.

  If in doing this you see that the files on which you are working on have been modified *upstream*,

    * move into your **local** branch,
    * merge the new master into your branch ``git merge master``,
    * resolve eventual conflicts
    * push to origin

  In this way, your pull request will be up-to-date with the master branch into which you want to merge your changes.
  If your changes are relatively small and `you know what you are doing <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>`_, you can use ``git rebase master``, instead of merging.

Open your pull-request
----------------------

When you or someone else open a pull-request which targets this issue, he/she
should:

- mirror the labels,
- if it's related to one ore more open issues, add in its description,

  - the phrase `Closes #X #Y ...` where X is the number associated to the issue(s) if any,
  - a reference to the issue, e.g. "as reported in #X" or similar

This will keep things clean and organized, so when you or
someone else land on the Projects page, the information is readily available
and updated.
