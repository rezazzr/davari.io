export interface QA {
  question: string;
  answer: string;
}

export const qas: QA[] = [
  {
    question: "How can I get an update when a new post comes out?",
    answer:
      'I post about my new post on this Twitter <a href="https://twitter.com/davari_reza" target="_blank">@davari_reza</a> account.',
  },
  {
    question: "What tool do you use for plotting?",
    answer:
      'I use <a href="http://draw.io/" target="_blank">draw.io</a>.',
  },
  {
    question: "Can I translate your posts to another language?",
    answer:
      'Yes, that would be my pleasure. But please <a href="mailto:davari.mreza@gmail.com">email</a> me in advance and please keep the original post link on top (rather than in tiny font at the end).',
  },
  {
    question:
      "I have papers on super relevant topics but you didn't include them?",
    answer:
      'It is challenging to write a comprehensive literature review. There are so many papers out there and I try my best to cover the essentials. I would be grateful if you could shoot me an <a href="mailto:davari.mreza@gmail.com">email</a> with pointers to interesting but missing papers. Cheers!',
  },
];
