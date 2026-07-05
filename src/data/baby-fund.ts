export interface PaymentMethod {
  /** e.g. "Zelle" */
  label: string;
  /** path to a small flag image (renders consistently across all OSes, unlike flag emoji) */
  flagSrc: string;
  /** alt text for the flag image */
  flagAlt: string;
  /** short audience line, e.g. "For friends & family in the US" */
  region: string;
  /** the email the transfer is sent to */
  email: string;
  /** optional helper line under the address (autodeposit note, etc.) */
  note?: string;
}

export interface UltrasoundPhoto {
  /**
   * Path under /public, e.g. "/assets/img/baby/scan-1.jpg".
   * Leave undefined to render a graceful "coming soon" placeholder tile.
   */
  src?: string;
  /** accessible description of the scan */
  alt: string;
  /** small caption shown under the photo */
  caption?: string;
  /**
   * Optional tiny base64 data URL (e.g. a 16px-wide version) for a true
   * blur-up while the full image loads. If omitted, a skeleton shimmer is
   * shown instead. Ask me to generate these when you add the real photos.
   */
  blurDataURL?: string;
}

export interface WayToHelp {
  emoji: string;
  text: string;
}

export const babyFund = {
  /** when the little one is due */
  dueDate: "the end of August",

  /**
   * Flip to `false` once the real Zelle / Interac emails below are filled in.
   * While `true`, a friendly "details coming soon" note is shown.
   */
  paymentsArePlaceholder: false,

  /** short copy for the highlighted card on the home page */
  homeNews: {
    badge: "Life Update",
    title: "We're expecting a little one! 🍼",
    blurb:
      "Due at the end of August. We skipped the registry, so here's the story, a few grayscale baby pics, and some easy ways to help if you'd like to.",
    ctaLabel: "Read our news",
  },

  hero: {
    eyebrow: "A little bit of news",
    title: "We're having a baby! 🍼",
    paragraphs: [
      "It still doesn't feel real to type out loud, but here we are: our first little human is due around the end of August, and we are somewhere between over-the-moon excited and mildly terrified (mostly the first one).",
      "A lot of you have been kind enough to ask about a registry or where to send a gift, so we put together this little corner of the internet to make it easy, and to say thank you before the sleep deprivation erases our manners.",
    ],
  },

  whyFund: {
    title: "Why a baby fund instead of a registry",
    paragraphs: [
      "Honestly? We don't have a registry. We're picking up most of the little stuff second-hand, since babies outgrow everything in about a week and it feels a little silly to buy it all brand new.",
      "The big-ticket things are a different story. The car seat, the stroller, all the gear with alarming price tags: those are the ones we're actually saving up for. And since most of our family is up in Canada while we're out here in Redmond, mailing boxes across the border gets complicated fast.",
      "So if you'd like to give something, a little contribution to the baby fund is genuinely the most helpful thing. It lets us put it exactly where it's needed, and it skips a customs form or two. No pressure at all, though; your excitement is already more than enough.",
    ],
  },

  paymentIntro:
    "Sending is quick. Just pick whichever works for where you are:",

  paymentMethods: [
    {
      label: "Zelle",
      flagSrc: "/assets/img/flag-us.png",
      flagAlt: "United States flag",
      region: "For friends & family in the US",
      email: "davari.reza@gmail.com",
      note: "Just send to this email and it will come straight to us.",
    },
    {
      label: "Interac e-Transfer",
      flagSrc: "/assets/img/flag-ca.png",
      flagAlt: "Canadian flag",
      region: "For friends & family in Canada",
      email: "reza.davari.93@gmail.com",
      note: "Autodeposit is on, so it lands automatically with no security question needed.",
    },
  ] as PaymentMethod[],

  placeholderNotice:
    "🚧 We're just finalizing the transfer details. The real Zelle and Interac info will be here any day now. Thanks for your patience!",

  waysToHelp: {
    title: "Other ways to help, because it takes a village",
    intro:
      "Money is lovely, but it genuinely does take a village. If you're the roll-up-your-sleeves type, here are a hundred non-money ways you could save our sanity in those first bleary-eyed months:",
    items: [
      {
        emoji: "🍲",
        text: "A home-cooked meal or something for the freezer, since we'll be running on zero sleep and pure gratitude.",
      },
      {
        emoji: "👶",
        text: "A cuddle shift, so we can nap, shower, or briefly remember our own names.",
      },
      {
        emoji: "🐕",
        text: "Dog walks or dog-sitting, since our pup is about to get a lot less attention than they're used to.",
      },
      {
        emoji: "🐈",
        text: "Keeping the cat and the dog entertained while we're buried under a mountain of tiny socks.",
      },
      {
        emoji: "☕",
        text: "An errand, a grocery run, a coffee drop-off. The small stuff is secretly the big stuff.",
      },
      {
        emoji: "💛",
        text: "Honestly, anything. It takes a village, and we're pretty lucky ours is a good one.",
      },
    ] as WayToHelp[],
  },

  giftsNote: {
    title: "Prefer to give a gift? Also wonderful.",
    body: "If you're more of a \"wrap something up\" person, that's completely lovely too. Second-hand, hand-me-downs, a onesie you couldn't resist: no wrong answers and zero pressure. It's the thought that gets us (that, and the hormones).",
  },

  thankYou: {
    title: "Thank you, really!",
    body: "Whether you send a few dollars, a lasagna, or just good vibes from across the border, it means the world to us. If you do send something, drop us a line so we can thank you properly, and so we know exactly who to blame for spoiling this kid.",
    ctaLabel: "Say hi 👋",
  },

  gallery: {
    title: "The first family photos",
    intro:
      "Just a peanut in grayscale for now, but we are already completely smitten.",
    photos: [
      {
        src: "/assets/img/baby/profile_face.png",
        alt: "Ultrasound profile view of the baby's face",
        caption: "That perfect little profile 🥹",
      },
      {
        src: "/assets/img/baby/sucking_thumb.png",
        alt: "Ultrasound of the baby sucking their thumb",
        caption: "Caught mid thumb-suck 👶",
      },
      {
        src: "/assets/img/baby/left_hand.png",
        alt: "Ultrasound of the baby's open left hand",
        caption: "A little left-hand wave 👋",
      },
    ] as UltrasoundPhoto[],
  },
} as const;
