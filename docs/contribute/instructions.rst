.. _instructions:

Instructions
============

.. contents::
    :local:
    :depth: 2

| These are some guidelines on how to contribute to *protopipe*.
| This of course makes sense only for the development branch, aka the *master*
  branch.

This is usually done is 4 steps:

1. you start using *protopipe*,
2. you find that either there is problem or *protopipe*
   is missing a feature that is important for your research,
3. you open an issue (or pull-request, if you already have a solution!)

Open an issue
-------------

| It is always preferable to open an issue first, in order to warn other 
  users/developers and possibly trigger a discussion.
| This will be useful to identify more precisely what needs to be done.

| If you are not able to do it, the administrators of the repository should **label
  your issue** depending on its nature.
| Labels are used to classify and prioritise issues within projects.

The labels normally used are quite self-explanatory, e.g.:

- bug
- fix
- wrong behaviour
- enhancement
- documentation
- dependency update
- summary

An issue can have multiple labels. You can propose new ones if needed.

Prepare and open a pull-request
-------------------------------

.. warning::

	It is assumed that you installed *protopipe* as a developer (:ref:`install-development`).

  1. update your **local** *master* branch with ``git pull upstream master``
  2. create and move to a new **local** branch from your **local** *master* with
     ``git checkout -b your_branch``
  3. develop inside it
  4. push it to *origin*, thereby creating a copy of your branch also there
  5. before pushing, please go through some checks (:ref:`beforepushing`)
  6. start a *pull request* using the web interface from *origin/your_branch*
     to *upstream/master*

    1. wait for an outcome
    2. if necessary, you can update or fix things in your branch because now
       everything is traced!
       (**local/your_branch** --> **origin/your_branch** --> **pull request**)

If your pull-request targets an issue, it should:

- have the same labels of that issue,
- if related to one ore more opened issues, its description should contain,

  - the phrase `Closes #X #Y ...` where X is the number associated to the issue(s) if any,
  - a reference to the issue, e.g. "as reported in #X ..." or similar.

This will keep things clean and organised, so when you or
someone else land on the Projects page, the information is readily available
and updated.

.. Note::

  If your developments take a relatively long time, consider to update
  periodically your **local** *master* branch.

  If while doing this you see that the files on which you are working have been
  modified *upstream*,

    * move into your **local** branch,
    * merge the new master into your branch ``git merge master``,
    * resolve eventual conflicts
    * push to origin

  In this way, your pull request will be up-to-date with the master branch into
  which you want to merge your changes.
  If your changes are relatively small and
  `you know what you are doing <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>`_,
  you can use ``git rebase master``, instead of merging.

Making your contribution visible
--------------------------------

Together with your changes, you should always check that,

- the email and name that you want to use is listed in the ``.mailmap``
- your name appears in the ``CODEOWNERS`` file according to your contribution

.. Note::
  
  | It can happen that, if you forget, the mantainer(s) will do it for you, but 
    please remember that it can be overlooked.
  | It is supposed to be a 
    responsibility of the authors of the pull request.
