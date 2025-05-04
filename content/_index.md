---
# Leave the homepage title empty to use the site title
title: ''
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Hi, I'm Prince!
      username: admin
  - block: resume-skills
    id: skills
    content:
      title: Skills & Hobbies
      # Note: `username` refers to the user's folder name in `content/authors/`
      username: admin
    design:
      show_skill_percentage: false
  # - block: experience
  #   content:
  #     # The user's folder name in `content/authors/`
  #     username: admin
  #   design:
  #     # Hugo date format
  #     date_format: 'January 2006'
  #     # Education or Experience section first?
  #     is_education_first: false                  
  # - block: collection
  #   id: projects
  #   content:
  #     title: Research Interests
  #     filters:
  #       folders:
  #         - project
  #   design:
  #     # Choose how many columns the section has. Valid values: '1' or '2'.
  #     columns: '2'
  #     view: card
  #     # For Showcase view, flip alternate rows?
  #     flip_alt_rows: false
  - block: collection
    id: featured-publications
    content:
      title: Featured publications
      filters:
        folders:
          - publication
        featured_only: true
    design:
      columns: '2'
      view: card
  - block: collection
    id: posts
    content:
      title: Recent posts
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 3
      # Filter on criteria
      filters:
        folders:
          - post
        author: ""
        category: ""
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: card
      columns: '2'
  # - block: collection
  #   id: other-publications
  #   content:
  #     title: Other publications
  #     # text: |-
  #     #   {{% callout note %}}
  #     #   Quickly discover relevant content by [filtering publications](./publication/).
  #     #   {{% /callout %}}
  #     filters:
  #       folders:
  #         - publication
  #       exclude_featured: true
  #   design:
  #     columns: '2'
  #     view: citation
  # - block: markdown
  #   content:
  #     title: Gallery
  #     subtitle: ''
  #     text: |-
  #       {{< gallery album="events" >}}
  #   design:
  #     columns: '2'

---
