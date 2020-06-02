.. _gitrepo:

The repository
==============

It is the place from which both users and developers can monitor the
status of the development of *protopipe*.

The best place where to start is the **projects** tab.
Here, as of today, there are 3 open projects:

- *Next release*,
- *Development of new algorithms*,
- *Bugs*.

All the projects do not point to specific deadlines, instead they are meant to
give a continuous overview about the current activities.

If what you had in mind is already covered there, you can participate,
otherwise you can open an issue yourself.

Next release
------------

This project collects all open issues and pull-requests that are related to the
work needed for releasing a new version of the pipeline.
It is organized in 4 sections:

- *Summary issues*, these are not "issues" in the real sense of the word,
  but rather GitHub issues that list enhancements which are all related to a particular subject
- *To Do*, these are open issues that should trigger pull-requests (some can be as simple as a question),
- *In progress*, pull-requests pushed by a user to the repository,
- *Review in progress*, one or some of the mantainers have started reviewing
  the pull-requests and/or discussing with the authors,
- *Reviewer approved*, the pull-request has been approved by the mantainers,
  but not yet merged into the master branch,
- *Done*, the pull-request has been accepted and merged; any issue linked to it
  (and likely appearing in the "To Do" section) will be automatically closed and will disappear.

At any point, if an issue or pull-request gets re-opened (maybe because there was
an error or an incompletness has been spotted) it will automatically reappear
in the corresponding section of this project.

Development of new algorithms
-----------------------------

By this we mean the development of new features that, even if not vital for the
next release, are still needed by the collaboration/observatory and can be
tested with this pipeline.
An example of this is the support for the divergent pointing technique developed
in ctapipe.

This project has the same structure of the "Next release" project and works in
the same way. In particular,

- relevant issues and pull-requests should be labelled as **additional tool**,
- the issues listed in the "Summary issues" column are expected to
  be isolated from each other, so each issue is a algorithm/subject itself.

Bugs
----

This is meant to be a tracker for bugs, but also for situations in which
the code works (so technically not a bug), but either we discovered a limitation
in performance or this has degraded for an unkown reason.

The project is divided in the following sections:

- *Needs triage*, collects all open issues tagged either **bug** or **wrong behaviour**
  that have not been classified by priority,
- *High priority*, open issues that previously needed triage, but that have been
  recognized to be fatal or urgent,
- *Low prioriy*, same but for issues related to non-urgent performance / non-fatal bugs,
- *In progress*, pull-requests opened to solve either of the prioritized issues
  of this project (could be under review or stale),
- *Done*, are closed issues or approved and merged pull-requests.
