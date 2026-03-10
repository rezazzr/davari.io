export interface Course {
  name: string;
  link: string;
  university: string;
  term: string;
}

export const courses: Course[] = [
  {
    name: "COMP-335: Introduction to Theoretical Computer Science",
    link: "/teaching/comp335",
    university: "Concordia University",
    term: "Fall 2019",
  },
  {
    name: "COMP-5361: Discrete Structures and Formal Languages",
    link: "/teaching/comp5361",
    university: "Concordia University",
    term: "Winter 2019",
  },
];
