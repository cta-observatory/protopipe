.. _gitrepo:

Code repository layout
======================

| The repository can be used to monitor the development status of *protopipe*.  
| It is organized in projects, which you can access from the
  `Projects tab <https://github.com/cta-observatory/protopipe/projects>`_.  
| They don't come with specific deadlines because they are meant to give a 
  continuous overview regardless of software versioning.

.. contents:: Current projects
    :local:
    :depth: 2

Pipeline features and enhancements
----------------------------------

| This project collects all open issues and pull-requests related to addition of
  new features or improvement of existing ones.
| It is organised in 4 sections:

- *Summary issues*, lists of issues all related to a particular subject,
- *To Do*, open issues that should trigger pull-requests (some can be as simple as a question),
- *In progress*, pull-requests pushed by a user to the repository,
- *Review in progress*, one or some of the maintainers started reviewing
  the pull-request(s) and/or discussing with the authors,
- *Reviewer approved*, the pull-request has been approved by the maintainers,
  but not yet merged into the master branch,
- *Done*, the pull-request has been accepted and merged; any linked issue
  in the "To Do" column will automatically disappear.

At any point, if an issue or pull-request gets re-opened it will automatically
reappear in the corresponding section of this project.

Pipeline applications
---------------------

| Features that, even if not vital for the
  next release, are still needed by some working groups and can be
  tested with this pipeline.
| An example of this is the support for the divergent pointing technique.

This project has the same structure of the *Pipeline features and enhancements*
project and works in the same way.

In particular,

- relevant issues and pull-requests should be labelled as ``pipeline applications``,
- the "Summary issues" are expected to
  be isolated from each other (each one referring to a different algorithm/project/application).

Bugs and wrong behaviours
-------------------------

A tracker for bugs, but also for situations in which
the code works (so technically not a bug), but either a limitation or degradation
in performance has been discovered.

The project is divided in the following sections:

- *Needs triage*, collects all open issues labelled either ``bug`` or ``wrong behaviour``
  that have not been classified by priority,
- *High priority*, open issues that previously needed triage, but that have been
  recognized to be fatal or urgent,
- *Low prioriy*, same but for issues related to non-urgent performance / non-fatal bugs,
- *In progress*, pull-requests opened to solve either of the prioritised issues
  of this project (could be under review or stale),
- *Done*, are closed issues or approved and merged pull-requests.

Maintenance
-----------

This project collects all open issues and pull-requests related to the
work needed to keep everything up-to-date and running in a safe development
environment, namely

- documentation
- unit-testing
- integration testing
- Continuous Integration (CI)
- release automation

It is organized as the *Pipeline features and enhancements* project
