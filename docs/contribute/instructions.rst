.. _instructions:

Instructions
============

.. contents::
    :local:
    :depth: 2

These are some guidelines on how to contribute to *protopipe* as a developer.

This is usually done is 4 steps:

1. you start using *protopipe*,
2. you find that either there is problem or *protopipe*
   is missing a feature that is important for your research,
3. you open an issue (doesn't matter if you have a solution ready to go: open an issue first).

Open an issue
-------------

| It is always preferable to open an issue first, in order to warn other 
  users/developers and possibly trigger a discussion.
| This will be useful to identify more precisely what needs to be done.

| If you are not able to do it, the administrators of the repository should **label
  your issue** depending on its nature.
| Labels are used to classify and prioritise issues within projects.

Some labels normally used are quite self-explanatory, e.g.:

- bug
- fix
- wrong behaviour
- enhancement
- documentation
- dependency update
- summary

An issue can have multiple labels. You can propose new ones if needed.

Prepare and open a pull-request (PR)
------------------------------------

.. warning::

	It is assumed that you installed *protopipe* as a developer (:ref:`install-development`).

Proceed as follows,

1. update your **local** *master* branch with ``git pull upstream master``
2. create and move to a new **local** branch from your **local** *master* with ``git checkout -b your_branch``
3. develop inside it
4. push it to *origin*, thereby creating a copy of your branch also there
5. before pushing, please go through the quality checks (:ref:`beforepushing`)
6. start a PR using the web interface from *origin/your_branch* to *upstream/master*

   1. wait for an outcome
   2. if necessary, you can update or fix things in your branch because now
      everything is traced! (**local/your_branch** --> **origin/your_branch** --> **pull request**)

Every PR **must**,

- be labeled (at least one label): this is needed by the release drafter to track contributions for the next release.
- have a meaningful description (regardless of the magnitude of the change),
- pass all required checks (if not, please convert it to a `Draft PR <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`__).

If your PR targets one or more open issues, it should:

- have the same labels of that issue,
- report in its description the phrase `Closes #X #Y ...` where X is the number associated to the issue(s).

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

  In this way, your PR will be up-to-date with the master branch into
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
  | It is supposed to be a responsibility of the authors of the pull request.

Creating a new release
----------------------

The project makes use of the `Release Drafter <https://github.com/apps/release-drafter>`__ GitHub App.  
A `release note draft <https://github.com/cta-observatory/protopipe/releases>`__
is created as soon as a new Pull-request is merged from the latest release.  
At each new merge it is updated and the template for it is defined in the `.github` folder at the root
of the project.

We follow semantic versioning and in particular `PEP440 <https://peps.python.org/pep-0440/>`__.

Each release will trigger a `Zenodo <https://zenodo.org/>`__ publication.  
After the release the DOI for the new release must be updated both on the README and documentation landing page.

Updating the Changelog
^^^^^^^^^^^^^^^^^^^^^^

The `CHANGELOG.rst` is stored at the root of the `docs` folder.
before making the release you need to make sure that the section for the new release is up-to-date
with the release draft and viceversa.

The project makes use of the `sphinx-issues <https://github.com/sloria/sphinx-issues#readme>` package
to link the GitHub issue tracker to the Sphinx-based documentation.  
Please, when you edit the changelog follow the formatting of existing releases.