// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "post-implementing-flash-attention-backward-pass-in-triton",
      
        title: "Implementing Flash Attention: Backward Pass in Triton",
      
      description: "In this follow-up post to Nathan Chen&#39;s Triton Flash Attention Kernel Walkthrough: The Forward Pass, we dive into gradient computation for queries, keys, and values in the backward pass.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/fa-bwd-triton/";
        
      },
    },{id: "news-start-to-work-as-teaching-assistant-for-ece-120-introduction-to-computing-with-prof-lin-qiu-prof-zuozhu-liu-and-prof-ujjal-bhowmik",
          title: 'Start to work as Teaching Assistant for ECE 120 Introduction to Computing, with...',
          description: "",
          section: "News",},{id: "news-start-to-work-as-teaching-assistant-for-ece-220-computer-systems-amp-amp-programming-with-prof-ujjal-bhowmik",
          title: 'Start to work as Teaching Assistant for ECE 220 Computer Systems &amp;amp;amp; Programming...',
          description: "",
          section: "News",},{id: "news-one-paper-accepted-by-the-second-cvpr-workshop-on-efficient-large-vision-models",
          title: 'One paper accepted by the second CVPR workshop on Efficient Large Vision Models...',
          description: "",
          section: "News",},{id: "news-one-paper-accepted-by-iccv-2025-see-you-in-hawaii",
          title: 'One paper accepted by ICCV 2025, see you in Hawaii!',
          description: "",
          section: "News",},{id: "news-one-paper-accepted-by-iccv-2025-findings",
          title: 'One paper accepted by ICCV 2025 Findings',
          description: "",
          section: "News",},{id: "news-video-mmlu-is-granted-outstanding-paper-award-by-iccv-2025-workshop-on-knowledge-intensive-multimodal-reasoning",
          title: 'Video-MMLU is granted Outstanding Paper Award by ICCV 2025 Workshop on Knowledge-Intensive Multimodal...',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-discord',
        title: 'Discord',
        section: 'Socials',
        handler: () => {
          window.open("https://discord.com/users/particle34", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%77%65%69%6C%69%78%75%32@%69%6C%6C%69%6E%6F%69%73.%65%64%75", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/Weili-0234", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/weili-xu-2a05662a7", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/William02319778", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
